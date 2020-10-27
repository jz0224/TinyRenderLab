cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME TinyRenderLab)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)

set (BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin/${PlatformName}")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(EXECUTABLE_OUTPUT_PATH ${BIN_DIR})

configure_file(config/config.in ${PROJECT_SOURCE_DIR}/config/config.out)
install(FILES "${PROJECT_SOURCE_DIR}/config/config.out"
  DESTINATION config)
configure_file(config/ROOT_PATH.h.in config/ROOT_PATH.h)

include_directories (
	${CMAKE_BINARY_DIR}/config
)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
MACRO(SOURCE_GROUP_BY_DIR source_files)
    if(MSVC)
        set(sgbd_cur_dir ${CMAKE_CURRENT_SOURCE_DIR})
        foreach(sgbd_file ${${source_files}})
            string(REGEX REPLACE ${sgbd_cur_dir}/\(.*\) \\1 sgbd_fpath ${sgbd_file})
            string(REGEX REPLACE "\(.*\)/.*" \\1 sgbd_group_name ${sgbd_fpath})
            string(COMPARE EQUAL ${sgbd_fpath} ${sgbd_group_name} sgbd_nogroup)
            string(REPLACE "/" "\\" sgbd_group_name ${sgbd_group_name})
			
            if(sgbd_nogroup)
                set(sgbd_group_name "\\")
            endif(sgbd_nogroup)
            source_group(${sgbd_group_name} FILES ${sgbd_file})
        endforeach(sgbd_file)
    endif(MSVC)
ENDMACRO(SOURCE_GROUP_BY_DIR)

add_subdirectory(src/Engine) 

#add_executable(${PROJECT_NAME} ${SRC_FILES} ${HEAD_FILES})