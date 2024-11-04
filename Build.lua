workspace "BST_SLAM"
   architecture "x64"
   configurations { "Debug", "Release" }
   startproject "App"

   -- Workspace-wide build options for MSVC
   filter "system:windows"
      buildoptions { "/EHsc", "/Zc:preprocessor", "/Zc:__cplusplus" }

OutputDir = "%{cfg.system}-%{cfg.architecture}/%{cfg.buildcfg}"

include "Lib/BST_SLAM.lua"
include "App/App.lua"