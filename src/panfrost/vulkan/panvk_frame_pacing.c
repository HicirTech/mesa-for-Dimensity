/*
 * Copyright © 2026
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>

#include "panvk_device.h"
#include "panvk_frame_pacing.h"

#include "util/list.h"
#include "util/macros.h"
#include "util/u_debug.h"
#include "vk_alloc.h"
#include "vk_device.h"
#include "vk_log.h"
#include "vk_queue.h"

#define NS_PER_SEC 1000000000ull
#define EMA_RESET_NS (250ull * 1000 * 1000)

struct panvk_frame_pacing_swapchain {
   struct list_head link;
   VkSwapchainKHR handle;
   uint64_t last_ns;
   uint64_t ema_ns;
   uint64_t refresh_ns;
   uint64_t last_actual_ns;
   VkTimeDomainKHR last_time_domain;
   uint64_t last_time_domain_id;
   VkPresentStageFlagsEXT last_actual_stage;
   bool has_last_actual;
   bool use_actual_timing;
   bool reset_pending;
};

struct panvk_frame_pacing {
   struct panvk_device *device;
   simple_mtx_t lock;
   struct list_head swapchains;
   uint64_t refresh_ns;
   bool debug;
   bool warned_oom;
   PFN_vkQueuePresentKHR queue_present;
   PFN_vkDestroySwapchainKHR destroy_swapchain;
   PFN_vkGetPastPresentationTimingEXT get_past_timing;
};

static bool
now_ns(uint64_t *ns)
{
   struct timespec ts;
   if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
      mesa_logw("PANVK_PACING: clock_gettime(CLOCK_MONOTONIC) failed: %s",
                strerror(errno));
      return false;
   }

   *ns = (uint64_t)ts.tv_sec * NS_PER_SEC + ts.tv_nsec;
   return true;
}

static void
update_ema(struct panvk_frame_pacing_swapchain *swapchain, uint64_t dt_ns)
{
   swapchain->ema_ns = swapchain->ema_ns - (swapchain->ema_ns >> 3) +
                       (dt_ns >> 3);
}

static struct panvk_frame_pacing_swapchain *
find_swapchain(struct panvk_frame_pacing *pacing, VkSwapchainKHR handle)
{
   list_for_each_entry(struct panvk_frame_pacing_swapchain, swapchain,
                       &pacing->swapchains, link) {
      if (swapchain->handle == handle)
         return swapchain;
   }

   return NULL;
}

static struct panvk_frame_pacing_swapchain *
get_swapchain(struct panvk_frame_pacing *pacing, VkSwapchainKHR handle)
{
   struct panvk_frame_pacing_swapchain *swapchain =
      find_swapchain(pacing, handle);
   if (swapchain)
      return swapchain;

   swapchain = vk_zalloc(&pacing->device->vk.alloc, sizeof(*swapchain),
                         alignof(struct panvk_frame_pacing_swapchain),
                         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!swapchain) {
      if (!pacing->warned_oom) {
         mesa_logw("PANVK_PACING: unable to allocate per-swapchain state");
         pacing->warned_oom = true;
      }
      return NULL;
   }

   swapchain->handle = handle;
   swapchain->refresh_ns = pacing->refresh_ns;
   list_addtail(&swapchain->link, &pacing->swapchains);
   return swapchain;
}

static void
remove_swapchain(struct panvk_frame_pacing *pacing, VkSwapchainKHR handle)
{
   struct panvk_frame_pacing_swapchain *swapchain;

   simple_mtx_lock(&pacing->lock);
   swapchain = find_swapchain(pacing, handle);
   if (swapchain)
      list_del(&swapchain->link);
   simple_mtx_unlock(&pacing->lock);

   vk_free(&pacing->device->vk.alloc, swapchain);
}

static uint64_t
actual_present_time(const VkPastPresentationTimingEXT *timing,
                    VkPresentStageFlagsEXT *stage_out)
{
   static const VkPresentStageFlagsEXT stage_order[] = {
      VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT,
      VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT,
      VK_PRESENT_STAGE_REQUEST_DEQUEUED_BIT_EXT,
      VK_PRESENT_STAGE_QUEUE_OPERATIONS_END_BIT_EXT,
   };

   if (!timing->pPresentStages)
      return 0;

   for (unsigned i = 0; i < ARRAY_SIZE(stage_order); i++) {
      for (uint32_t j = 0; j < timing->presentStageCount; j++) {
         if (timing->pPresentStages[j].stage == stage_order[i] &&
             timing->pPresentStages[j].time != 0) {
            *stage_out = timing->pPresentStages[j].stage;
            return timing->pPresentStages[j].time;
         }
      }
   }

   *stage_out = 0;
   return 0;
}

static void
observe_present_timings(struct panvk_frame_pacing *pacing,
                        VkSwapchainKHR handle,
                        uint32_t timing_count,
                        const VkPastPresentationTimingEXT *timings)
{
   simple_mtx_lock(&pacing->lock);
   struct panvk_frame_pacing_swapchain *swapchain =
      find_swapchain(pacing, handle);
   if (!swapchain) {
      simple_mtx_unlock(&pacing->lock);
      return;
   }

   for (uint32_t i = 0; i < timing_count; i++) {
      const VkPastPresentationTimingEXT *timing = &timings[i];
      VkPresentStageFlagsEXT stage;
      const uint64_t present_ns = actual_present_time(timing, &stage);
      if (!present_ns)
         continue;

      const bool changed_domain =
         swapchain->has_last_actual &&
         (timing->timeDomain != swapchain->last_time_domain ||
          timing->timeDomainId != swapchain->last_time_domain_id ||
          stage != swapchain->last_actual_stage);
      if (!swapchain->has_last_actual || changed_domain) {
         swapchain->last_actual_ns = present_ns;
         swapchain->last_time_domain = timing->timeDomain;
         swapchain->last_time_domain_id = timing->timeDomainId;
         swapchain->last_actual_stage = stage;
         swapchain->has_last_actual = true;
         if (changed_domain) {
            swapchain->ema_ns = 0;
            swapchain->reset_pending = true;
         }
         continue;
      }

      if (present_ns <= swapchain->last_actual_ns)
         continue;

      const uint64_t dt_ns = present_ns - swapchain->last_actual_ns;
      swapchain->last_actual_ns = present_ns;
      if (!swapchain->use_actual_timing) {
         swapchain->ema_ns = 0;
         swapchain->use_actual_timing = true;
      }
      if (dt_ns > EMA_RESET_NS) {
         swapchain->ema_ns = 0;
         swapchain->reset_pending = true;
      } else {
         update_ema(swapchain, dt_ns);
      }
   }
   simple_mtx_unlock(&pacing->lock);
}

static VkResult VKAPI_CALL
panvk_frame_pacing_get_past_timing(
   VkDevice _device,
   const VkPastPresentationTimingInfoEXT *info,
   VkPastPresentationTimingPropertiesEXT *properties)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   struct panvk_frame_pacing *pacing = device->frame_pacing;
   if (!pacing || !pacing->get_past_timing)
      return VK_ERROR_EXTENSION_NOT_PRESENT;

   const uint32_t capacity = properties->presentationTimingCount;
   VkPastPresentationTimingEXT *timings = properties->pPresentationTimings;
   VkResult result = pacing->get_past_timing(_device, info, properties);

   if ((result == VK_SUCCESS || result == VK_INCOMPLETE) && timings) {
      observe_present_timings(pacing, info->swapchain,
                              MIN2(properties->presentationTimingCount,
                                   capacity),
                              timings);
   }

   return result;
}

static void VKAPI_CALL
panvk_frame_pacing_destroy_swapchain(
   VkDevice _device, VkSwapchainKHR swapchain,
   const VkAllocationCallbacks *allocator)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   struct panvk_frame_pacing *pacing = device->frame_pacing;

   remove_swapchain(pacing, swapchain);
   pacing->destroy_swapchain(_device, swapchain, allocator);
}

static void
sleep_until(uint64_t deadline_ns)
{
   const struct timespec deadline = {
      .tv_sec = deadline_ns / NS_PER_SEC,
      .tv_nsec = deadline_ns % NS_PER_SEC,
   };

   int ret;
   do {
      ret = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, NULL);
   } while (ret == EINTR);

   if (ret != 0)
      mesa_logw("PANVK_PACING: clock_nanosleep failed: %s", strerror(ret));
}

static VkResult VKAPI_CALL
panvk_frame_pacing_queue_present(VkQueue _queue,
                                 const VkPresentInfoKHR *present_info)
{
   struct vk_queue *queue = vk_queue_from_handle(_queue);
   struct panvk_device *device = to_panvk_device(queue->base.device);
   struct panvk_frame_pacing *pacing = device->frame_pacing;
   uint64_t now;

   if (!now_ns(&now))
      return pacing->queue_present(_queue, present_info);

   uint64_t deadline_ns = now;
   uint64_t max_ema_ns = 0;
   uint64_t max_delay_ns = 0;
   bool over_cap = false;
   bool reset = false;
   bool actual_source = false;

   simple_mtx_lock(&pacing->lock);
   for (uint32_t i = 0; i < present_info->swapchainCount; i++) {
      struct panvk_frame_pacing_swapchain *swapchain =
         get_swapchain(pacing, present_info->pSwapchains[i]);
      if (!swapchain)
         continue;

      if (swapchain->last_ns == 0) {
         swapchain->last_ns = now;
         continue;
      }

      if (swapchain->reset_pending) {
         swapchain->ema_ns = 0;
         swapchain->reset_pending = false;
         swapchain->last_ns = now;
         reset = true;
         continue;
      }

      if (swapchain->use_actual_timing) {
         actual_source = true;
      } else {
         const uint64_t dt_ns =
            now > swapchain->last_ns ? now - swapchain->last_ns : 0;
         if (dt_ns > EMA_RESET_NS) {
            swapchain->ema_ns = 0;
            swapchain->last_ns = now;
            reset = true;
            continue;
         }
         update_ema(swapchain, dt_ns);
      }

      max_ema_ns = MAX2(max_ema_ns, swapchain->ema_ns);
      const uint64_t interval_ns =
         MAX2(swapchain->ema_ns, swapchain->refresh_ns);
      const uint64_t target_ns = swapchain->last_ns + interval_ns;
      const uint64_t delay_ns = target_ns > now ? target_ns - now : 0;
      if (delay_ns > swapchain->refresh_ns) {
         over_cap = true;
         continue;
      }

      if (delay_ns > max_delay_ns) {
         max_delay_ns = delay_ns;
         deadline_ns = target_ns;
      }
   }
   simple_mtx_unlock(&pacing->lock);

   const char *decision = max_delay_ns ? "sleep" :
                         over_cap ? "over-cap" :
                         reset ? "reset" : "on-time";
   if (pacing->debug) {
      mesa_logi("PANVK_PACING: ema=%" PRIu64 " ns delay=%" PRIu64
                " ns source=%s decision=%s",
                max_ema_ns, max_delay_ns,
                actual_source ? "present-timing" : "cpu", decision);
   }

   if (max_delay_ns)
      sleep_until(deadline_ns);

   VkResult result = pacing->queue_present(_queue, present_info);

   uint64_t present_ns;
   if (now_ns(&present_ns)) {
      simple_mtx_lock(&pacing->lock);
      for (uint32_t i = 0; i < present_info->swapchainCount; i++) {
         struct panvk_frame_pacing_swapchain *swapchain =
            find_swapchain(pacing, present_info->pSwapchains[i]);
         if (swapchain)
            swapchain->last_ns = present_ns;
      }
      simple_mtx_unlock(&pacing->lock);
   }

   return result;
}

VkResult
panvk_frame_pacing_init(struct panvk_device *device)
{
   if (!debug_get_bool_option("PANVK_PACING", true))
      return VK_SUCCESS;

   const int64_t hz = debug_get_num_option("PANVK_PACING_HZ", 60);
   if (hz <= 0 || hz > 1000) {
      mesa_logw("PANVK_PACING_HZ=%" PRId64 " is outside 1..1000; using 60",
                hz);
   }
   const uint64_t refresh_ns = NS_PER_SEC /
                               (hz > 0 && hz <= 1000 ? hz : 60);

   if (!device->vk.dispatch_table.QueuePresentKHR ||
       !device->vk.dispatch_table.DestroySwapchainKHR) {
      mesa_logw("PANVK_PACING is enabled but WSI present dispatch is unavailable");
      return VK_SUCCESS;
   }

   struct panvk_frame_pacing *pacing =
      vk_zalloc(&device->vk.alloc, sizeof(*pacing),
                alignof(struct panvk_frame_pacing),
                VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!pacing)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   pacing->device = device;
   pacing->refresh_ns = refresh_ns;
   pacing->debug = debug_get_bool_option("PANVK_PACING_DEBUG", false);
   pacing->queue_present = device->vk.dispatch_table.QueuePresentKHR;
   pacing->destroy_swapchain = device->vk.dispatch_table.DestroySwapchainKHR;
   if (device->vk.enabled_extensions.EXT_present_timing)
      pacing->get_past_timing =
         device->vk.dispatch_table.GetPastPresentationTimingEXT;

   simple_mtx_init(&pacing->lock, mtx_plain);
   list_inithead(&pacing->swapchains);

   device->frame_pacing = pacing;
   device->vk.dispatch_table.QueuePresentKHR =
      panvk_frame_pacing_queue_present;
   device->vk.dispatch_table.DestroySwapchainKHR =
      panvk_frame_pacing_destroy_swapchain;
   if (pacing->get_past_timing)
      device->vk.dispatch_table.GetPastPresentationTimingEXT =
         panvk_frame_pacing_get_past_timing;

   return VK_SUCCESS;
}

void
panvk_frame_pacing_finish(struct panvk_device *device)
{
   struct panvk_frame_pacing *pacing = device->frame_pacing;
   if (!pacing)
      return;

   device->vk.dispatch_table.QueuePresentKHR = pacing->queue_present;
   device->vk.dispatch_table.DestroySwapchainKHR = pacing->destroy_swapchain;
   if (pacing->get_past_timing)
      device->vk.dispatch_table.GetPastPresentationTimingEXT =
         pacing->get_past_timing;

   list_for_each_entry_safe(struct panvk_frame_pacing_swapchain, swapchain,
                            &pacing->swapchains, link) {
      list_del(&swapchain->link);
      vk_free(&device->vk.alloc, swapchain);
   }

   simple_mtx_destroy(&pacing->lock);
   vk_free(&device->vk.alloc, pacing);
   device->frame_pacing = NULL;
}
