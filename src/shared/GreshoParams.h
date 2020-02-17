#pragma once

#include "shared/real_type.h"
#include "utils/config/ConfigMap.h"

struct GreshoParams
{
    // gresho problem parameters
    real_t gresho_center_x;
    real_t gresho_center_y;
    real_t gresho_mach;

    GreshoParams(ConfigMap& configMap)
    {
        real_t xmin = configMap.getFloat("mesh", "xmin", 0.0);
        real_t ymin = configMap.getFloat("mesh", "ymin", 0.0);

        real_t xmax = configMap.getFloat("mesh", "xmax", 1.0);
        real_t ymax = configMap.getFloat("mesh", "ymax", 1.0);

        gresho_center_x = configMap.getFloat("gresho","center_x", (xmin+xmax)/2);
        gresho_center_y = configMap.getFloat("gresho","center_y", (ymin+ymax)/2);
        gresho_mach     = configMap.getFloat("gresho", "mach", 0.1);
    }
}; // struct GreshoParams
