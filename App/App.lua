project "App"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++20"
   targetdir "Bin/%{cfg.buildcfg}"
   staticruntime "off"

   files { "Source/**.hpp", "Source/**.cpp" }

   includedirs
   {
      "../Vendor/OpenCV/include",
      "../Vendor/DepthAI/include",
      "../Lib/Source",

      "Source"
   }

   links
   {
      "../Vendor/OpenCV/lib/opencv_world4100.lib",
      "../Vendor/DepthAI/lib/depthai-core.lib",
      "../Vendor/DepthAI/lib/depthai-opencv.lib"
   }

   targetdir ("../Bin/" .. OutputDir .. "/%{prj.name}")
   objdir ("../Bin/Intermediates/" .. OutputDir .. "/%{prj.name}")

   filter "system:windows"
       systemversion "latest"
       defines { "WINDOWS" }

   filter "configurations:Debug"
       defines { "DEBUG" }
       runtime "Debug"
       symbols "On"

   filter "configurations:Release"
       defines { "RELEASE" }
       runtime "Release"
       optimize "On"
       symbols "On"
