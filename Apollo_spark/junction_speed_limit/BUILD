load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("//tools:apollo.bzl", "cyber_plugin_description")
load("//tools:apollo_package.bzl", "apollo_cc_library", "apollo_package", "apollo_plugin")
load("//tools/proto:proto.bzl", "proto_library")
load("//tools:cpplint.bzl", "cpplint")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "junction_speed_limit_files",
    srcs = glob([
        "conf/**",
    ]),
)

apollo_plugin(
    name = "libjunction_speed_limit.so",
    srcs = [
        "junction_speed_limit.cc",
    ],
    hdrs = [
        "junction_speed_limit.h",
    ],
    description = ":plugins.xml",
    copts = ["-DMODULE_NAME=\\\"planning\\\""],
    deps = [
        "//modules/common_msgs/planning_msgs:planning_cc_proto",
        "//modules/map:apollo_map",
        "//cyber",
        "//modules/planning/tasks/junction_speed_limit/proto:junction_speed_limit_proto",
        "//modules/planning/planning_interface_base:apollo_planning_planning_interface_base",

    ],
)

apollo_package()

cpplint()
