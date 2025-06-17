
#pragma once

#include <memory>
#include "cyber/plugin_manager/plugin_manager.h"
#include <vector>

// proto
#include "modules/planning/tasks/junction_speed_limit/proto/junction_speed_limit.pb.h"

#include "modules/map/pnc_map/path.h"
#include "modules/common_msgs/routing_msgs/routing.pb.h"

#include "modules/common/status/status.h"
#include "modules/planning/planning_interface_base/task_base/task.h"
#include "modules/planning/planning_interface_base/task_base/common/path_generation.h"

namespace apollo {
namespace planning {

class JunctionSpeedLimit : public PathGeneration {
public:
    bool Init(
            const std::string& config_dir,
            const std::string& LaneWaypointname,
            const std::shared_ptr<DependencyInjector>& injector);

    apollo::common::Status Process(Frame* const frame, ReferenceLineInfo* const reference_line_info);

    JunctionSpeedLimitConfig config_;
};

CYBER_PLUGIN_MANAGER_REGISTER_PLUGIN(apollo::planning::JunctionSpeedLimit, Task)

}  // namespace planning
}  // namespace apollo
