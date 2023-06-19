#!/bin/bash

# Set the paths as environment variables
export CELLPACK_PATH="$HOME"/HemoCell/tools/packCells
export CELLPACK_BIN=."$HOME"/HemoCell/tools/packCells/build/packCells

# Clone the repository
if ! command -v packCells &> /dev/null
then
    echo "cellpack could not be found, installing now"

    git clone https://github.com/UvaCsl/HemoCell.git
    # Change to the Cellpack directory
    cd $CELLPACK_PATH

    # Make a new directory and change into it
    mkdir build 
    cd build

    # Run cmake and make
    cmake ../
    make

    # Finally, go back to the performance analysis directory
    cd "$HOME"

fi

packCells -h
