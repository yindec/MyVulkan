set GLSLC="D:/WorkSpace/VulkanSDK/1.3.261.1/Bin/glslc.exe"
set SOURCE_DIR="./"

for /d %%d in (%SOURCE_DIR%\*) do (
    for %%f in (%%d\*.vert %%d\*.frag %%d\*.comp) do (
        %GLSLC% "%%f" -o "%%f.spv"
    )
)

@REM pause