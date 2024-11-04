project "BST_SLAM"
   kind "StaticLib"
   language "C++"
   cppdialect "C++20"
   targetdir "Bin/%{cfg.buildcfg}"
   staticruntime "off"

   files { "Source/**.hpp", "Source/**.cpp" }

   includedirs
   {
      "Source",
      "../Vendor/OpenCV/include"
   }

   links
   {
      "../Vendor/OpenCV/lib/opencv_world4100.lib"
   }

   targetdir ("../Bin/" .. OutputDir .. "/%{prj.name}")
   objdir ("../Bin/Intermediates/" .. OutputDir .. "/%{prj.name}")

   filter "system:windows"
       systemversion "latest"
       defines { }

   filter "configurations:Dist"
       defines { "DIST" }
       runtime "Release"
       optimize "On"
       symbols "Off"