add_subdirectory(camerautils)
add_subdirectory(auxiliary)
add_subdirectory(fusion)

# FIND_PACKAGE(QGLViewer REQUIRED)
# FIND_PACKAGE(Qt REQUIRED)
find_package(Qt5 COMPONENTS Core Gui OpenGL Widgets Gui Qml Xml 
    Concurrent
    Core
    DBus
    Designer
    Gui
    Help
    Network
    OpenGL
    OpenGLExtensions
    PrintSupport
    Qml
    Quick
    QuickTest
    QuickWidgets
    Sql
    Test
    UiPlugin
    UiTools
    WebKit
    WebKitWidgets
    Widgets
    Xml
    REQUIRED)
    include_directories(${Qt5Widgets_INCLUDE_DIRS}
    # "/usr/include/QGLViewer"
)
message(${Qt5Widgets_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/libQGLViewer-2.7.2/QGLViewer)
link_directories(${PROJECT_SOURCE_DIR}/libQGLViewer-2.7.2/QGLViewer)
message(${PROJECT_SOURCE_DIR}/libQGLViewer-2.7.2/QGLViewer)
INCLUDE_DIRECTORIES(${QT5_INCLUDES})
INCLUDE_DIRECTORIES(${Qt5Svg_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Qt5OpenGL_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Qt5Concurrent_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Qt5Xml_INCLUDE_DIRS})
# message(${QGLViewer_LIBRARIES})
QT5_WRAP_CPP(onlinefusion_HEADERS_MOC onlinefusionviewer.hpp)
add_executable(onlinefusion
onlinefusionviewer_main.cpp
onlinefusionviewer.cpp
${onlinefusion_HEADERS_MOC}
)
target_compile_options(onlinefusion PRIVATE -fPIC)
target_link_libraries(onlinefusion
    geometryfusion_mipmap_cpu geometryfusion_aos
    camerautils
    auxiliary
    ${OpenCV_LIBS}
    ${QGLViewer_LIBRARIES} ${QT5_LIBRARIES} ${QGLVIEWER_LIBRARY} 
    libQGLViewer-qt5.so 
    Qt5::Core Qt5::Gui Qt5::Qml Qt5::Quick ${Qt5Widgets_LIBRARIES}
    GL GLU glut GLEW
    boost_thread
    -fPIC
    -fPIE
    boost_system
)