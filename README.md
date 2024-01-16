# PlusML

PlusML is a C++ library that aims to provide implementations of common machine learning algorithms with clean API.

### Table of contents

* [Getting Started](#getting-started)
* [Usage](#usage)
* [Docs](#docs)
* [Dependencies](#dependencies)
* [Testing](#testing)
* [License](#license)

### Getting Started

The PlusML library can be installed via `git submodule`.

Navigate to the project's directory and run

```shell
git submodule add https://github.com/hashlag/plusml
```

then initialize and update submodules

```shell
git submodule update --init --recursive
```

Now you can compose your main `CMakeLists.txt` to add PlusML to your project.

Example for Windows systems:

```cmake
cmake_minimum_required(VERSION 3.27)
project(myproject)

set(CMAKE_CXX_STANDARD 20)

# Add PlusML directory to the project
add_subdirectory(plusml)

add_executable(myproject main.cpp)

# Link PlusML to your executable
target_link_libraries(myproject PRIVATE plusml)

# Include PlusML headers
target_include_directories(myproject PRIVATE plusml/include)

# Copy plusml.dll to the directory with your project's executable after building
add_custom_command(TARGET myproject POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE_DIR:plusml>/plusml.dll
        $<TARGET_FILE_DIR:myproject>
        COMMENT "Copying plusml.dll"
)
```

After configuring CMake you can build your project via CMake CLI or your favourite IDE tools.

Manual building may look like this:

```shell
cmake -S . -B ./build-dir
```

```shell
cmake --build ./build-dir
```

Where `.` is your project's directory.

### Usage

After installing the library you can use provided algorithms by including corresponding headers.

API information is provided in [docs](#docs).

Basic example:

```c++
#include <iostream>
#include <PlusML/linear_regression.h>

int main() {
    plusml::LinearRegression model(2);

    std::cout << model.Parameters();

    return 0;
}
```

Returns

```
0
0
0
```

since we have two features, bias is enabled by default and parameters are initialized with zeros.

### Docs

PlusML provides auto-generated docs via [Doxygen](https://www.doxygen.nl/).

Feel free to explore it:

* [Main Page](https://hashlag.github.io/plusml/)
* [Class Hierarchy](https://hashlag.github.io/plusml/hierarchy.html)
* [Table of Contents](https://hashlag.github.io/plusml/classes.html)

### Testing

Tests for all implemented algorithms are placed at the `test/` directory along with testcase generators.

CMake configuration for tests is also provided, just uncomment the corresponding section in `CMakeLists.txt` according to [the hints](https://github.com/hashlag/plusml/blob/7f3f9ce91ae6106a3324ecbeefc53a280a8e7b4b/CMakeLists.txt#L29).

### Dependencies

PlusML depends on [Eigen linear algebra library](https://eigen.tuxfamily.org/index.php?title=Main_Page) which is installed as a `git submodule` and does not require any dependencies other than the C++ standard library.

We also use [GoogleTest](https://github.com/google/googletest) for testing purposes but PlusML does not require it to run.

### License

PlusML is licensed under [the MIT License](https://raw.githubusercontent.com/hashlag/plusml/main/LICENSE).