/*
 * Copyright © 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_FRAME_PACING_H
#define PANVK_FRAME_PACING_H

struct panvk_device;

VkResult panvk_frame_pacing_init(struct panvk_device *device);
void panvk_frame_pacing_finish(struct panvk_device *device);

#endif
