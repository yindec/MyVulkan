set GLSLC="D:/WorkSpace/VulkanSDK/1.3.261.1/Bin/glslc.exe"

for %%f in (./\*.vert ./\*.frag ./\*.comp) do (
    %GLSLC% "%%f" -o "%%f.spv"
)

@REM pause