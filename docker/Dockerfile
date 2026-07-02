FROM nvcr.io/nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y ca-certificates git pkgconf ninja-build ccache libssl-dev wget libgoogle-glog-dev libglib2.0-dev && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0-linux-x86_64.sh && \
    chmod +x cmake-3.24.0-linux-x86_64.sh && \
    ./cmake-3.24.0-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm -rf /tmp/*

# Nvidia
RUN cd /tmp && \
    git clone -b n10.0.26.2 --depth=1 https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make && make install && \
    rm -rf /tmp/*
RUN apt update && \
    apt install -y nasm libvpx-dev && \
    rm -rf /var/lib/apt/lists/*
RUN cd /tmp && \
    git clone -b n4.4.2 --depth=1 https://github.com/FFmpeg/FFmpeg.git && cd FFmpeg && \
    ./configure \
        --prefix=/usr/local/x86_64-nvidia-gnu \
        --disable-debug \
        --disable-programs \
        --enable-shared \
        --enable-openssl \
        --enable-cuda \
        --enable-cuvid \
        --enable-nvenc \
        --enable-libnpp \
        --enable-nonfree \
        --enable-optimizations \
        --extra-libs="-lpthread -lm" \
        --extra-cflags="-I/usr/local/cuda-11.4/include" \
        --extra-ldflags="-L/usr/local/cuda-11.4/lib64" && \
    make -j && make install && \
    rm -rf /tmp/*
RUN apt update && \
    apt install -y libjpeg-dev libpng-dev libtiff-dev libopencore-amrnb-dev libopencore-amrwb-dev \
        libtbb-dev libatlas-base-dev && \
    rm -rf /var/lib/apt/lists/*
RUN cd /tmp && \
    git clone -b 4.5.5 --depth=1 https://github.com/opencv/opencv_contrib.git && \
    git clone -b 4.5.5 --depth=1 https://github.com/opencv/opencv.git && cd opencv && \
    cmake -S . -Bbuild -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/x86_64-nvidia-gnu \
        -DCMAKE_BUILD_TYPE=Release -DWITH_TBB=ON -DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=1 \
        -DWITH_CUBLAS=1 -DWITH_CUDA=ON -DBUILD_opencv_cudacodec=OFF -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON \
        -DCUDA_ARCH_BIN=7.5 -DWITH_QT=OFF -DWITH_OPENGL=ON -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DINSTALL_C_EXAMPLES=OFF -DBUILD_EXAMPLES=OFF -DWITH_FFMPEG=OFF && \
    cmake --build build/ --target install && \
    rm -rf /tmp/*
RUN apt update && \
    apt install -y libnvinfer8=8.2.5-1+cuda11.4 libnvonnxparsers8=8.2.5-1+cuda11.4 \
        libnvparsers8=8.2.5-1+cuda11.4 libnvinfer-plugin8=8.2.5-1+cuda11.4 \
        libnvinfer-dev=8.2.5-1+cuda11.4 libnvonnxparsers-dev=8.2.5-1+cuda11.4 \
        libnvparsers-dev=8.2.5-1+cuda11.4 libnvinfer-plugin-dev=8.2.5-1+cuda11.4 && \
    apt-mark hold libnvinfer8 libnvonnxparsers8 libnvparsers8 libnvinfer-plugin8 \
        libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev && \
    rm -rf /var/lib/apt/lists/*

# Intel
RUN cd /tmp && \
    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB --no-check-certificate && \
    gpg --dearmor -o /usr/share/keyrings/intel-archive.gpg GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    rm -rf /tmp/*
RUN echo "deb [signed-by=/usr/share/keyrings/intel-archive.gpg] https://apt.repos.intel.com/openvino/2022 focal main" | tee /etc/apt/sources.list.d/intel-openvino-2022.list && \
    apt update && \
    apt install -y openvino-2022.1.0 openvino-opencv && \
    rm -rf /var/lib/apt/lists/*
RUN apt update && \
    apt install -y libdrm-dev autoconf libtool pkg-config && \
    rm -rf /var/lib/apt/lists/*
RUN cd /tmp && \
    git clone -b 2.15.0 --depth=1 https://github.com/intel/libva.git && cd libva && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local/x86_64-intel-gnu && \
    make -j12 && make install && \
    rm -rf /tmp/*
RUN cd /tmp && \
    git clone -b intel-gmmlib-22.1.7 --depth=1 https://github.com/intel/gmmlib.git && cd gmmlib && \
    cmake -S . -Bbuild -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/x86_64-intel-gnu && \
    cmake --build build/ --target install && \
    rm -rf /tmp/*
RUN cd /tmp && \
    git clone -b intel-media-22.5.2 --depth=1 https://github.com/intel/media-driver.git && cd media-driver && \
    cmake -S . -Bbuild -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/x86_64-intel-gnu \
        -DPKG_CONFIG_USE_CMAKE_PREFIX_PATH=ON -DCMAKE_PREFIX_PATH=/usr/local/x86_64-intel-gnu && \
    cmake --build build/ --target install && \
    rm -rf /tmp/*
RUN cd /tmp && \
    git clone -b n4.4.2 --depth=1 https://github.com/FFmpeg/FFmpeg.git && cd FFmpeg && \
    PKG_CONFIG_PATH="/usr/local/x86_64-intel-gnu/lib/pkgconfig" \
    ./configure \
        --prefix=/usr/local/x86_64-intel-gnu \
        --disable-debug \
        --disable-programs \
        --enable-shared \
        --enable-openssl \
        --enable-vaapi \
        --enable-nonfree \
        --enable-optimizations \
        --extra-libs="-lpthread -lm" && \
    make -j && make install && \
    rm -rf /tmp/*

# Docker
RUN apt update && \
    apt install -y curl lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt update && \
    apt -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/*

# User
RUN apt update && \
    apt install -y sudo && \
    rm -rf /var/lib/apt/lists/*
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} ai && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash ai && \
    usermod -aG sudo ai && \
    echo 'ai:111111' | chpasswd && \
    mkdir -p /workspace && chown ai /workspace && \
    gpasswd -a ai docker
RUN apt clean autoclean && apt autoremove --yes
USER ai

# Env
ENV PATH=/usr/local/cuda-11.4/bin:/usr/local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/usr/local/lib:$LD_LIBRARY_PATH
RUN echo "export C_INCLUDE_PATH=/usr/local/include:${C_INCLUDE_PATH:-}" >> ~/.bashrc && \
    echo "export CPLUS_INCLUDE_PATH=/usr/local/include:${CPLUS_INCLUDE_PATH:-}" >> ~/.bashrc
WORKDIR /workspace

RUN ["/bin/bash"]
