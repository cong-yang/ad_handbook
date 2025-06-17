#include <memory>
#include <unordered_map>

#include "modules/planning/tasks/junction_speed_limit/junction_speed_limit.h"

#include "modules/planning/planning_base/gflags/planning_gflags.h"

namespace apollo {
namespace planning {
using apollo::common::Status;
using apollo::hdmap::PathOverlap;

bool JunctionSpeedLimit::Init(
        const std::string& config_dir,
        const std::string& LaneWaypointname,
        const std::shared_ptr<DependencyInjector>& injector) {
    if (!Task::Init(config_dir, LaneWaypointname, injector)) {
        return false;
    }
    return Task::LoadConfig<JunctionSpeedLimitConfig>(&config_);
}

apollo::common::Status JunctionSpeedLimit::Process(Frame* const frame, ReferenceLineInfo* const reference_line_info) {
    AINFO << "JunctionSpeedLimit::Process";
    ReferenceLine* reference_line = reference_line_info->mutable_reference_line();
    const std::vector<PathOverlap>& crosswalk_overlaps
            = reference_line_info->reference_line().map_path().crosswalk_overlaps();
    for (const auto& crosswalk_overlap : crosswalk_overlaps) {
        reference_line->AddSpeedLimit(
                crosswalk_overlap.start_s - config_.forward_buffer(),
                crosswalk_overlap.end_s + config_.backward_buffer(),
                config_.limit_speed());
    }
    return Status::OK();
}

}  // namespace planning
}  // namespace apollo
