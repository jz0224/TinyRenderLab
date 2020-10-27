set(PROJECT_NAME RenderLab_Engine)
project(${PROJECT_NAME})

FILE(GLOB PROJECT_SOURCE_FILES ${PROJECT_SOURCE_FILES}
        Basic/*.h
        Basic/*.cpp
        Core/*.h
        Core/*.cpp
        Material/*.h
        Material/*.cpp
        Standard/*.h
        Standard/*.cpp
)

include_directories (
        Basic
        Core
        Material
        Standard
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_FILES})

SOURCE_GROUP_BY_DIR(PROJECT_SOURCE_FILES)

SET_PROPERTY(TARGET ${PROJECT_NAME} PROPERTY FOLDER TinyRenderLab)