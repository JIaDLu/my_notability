## Docker

### docker Hub

#### 介绍：

docker官方维护了一个公共仓库Docker Hub，其中已经包括了数量超过 2,650,000 的镜像。大部分需求都可以通过在 Docker Hub 中直接下载镜像来实现。

`把他类比 github，docker 是存放镜像，github 是存放代码的。`

这样自己制作的镜像就可以把它推送（docker push）到 DockerHub，要使用的时候直接拉取（docker pull），让开发更加灵活。

#### 使用docker hub

docker hub存放着docker镜像 及 其组件 的所有资源。docker hub可以帮助你与同事之间协作，并获得功能完整的docker。



![image-20240906105519168](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202409061055197.png)

![image-20240906105604143](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202409061056164.png)

https://hub.docker.com/_/centos/tags?page_size=&ordering=&name=latest

![**image-20240906105408208**](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202409061054382.png)



![image-20240906105656286](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202409061056305.png)



![image-20240906105741526](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202409061057547.png)



![image-20240906111248627](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202409061112688.png)



![image-20240906111308004](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202409061113031.png)



### docker与虚拟机的区别：

1，运行方式：

虚拟机：虚拟机是通过在物理机上模拟出完整的硬件环境，然后安装一个完整的操作系统。每个虚拟机都有自己的一套操作系统、内核、应用程序，`比较重，需要更多的资源`

docker：`更轻量，它不需要模拟硬件`，基于宿主机的操作系统(例如 我是macos)，那么它只隔离应用和所需要的依赖。我拉了一个docker镜像，它直接共享宿主机的内核，启动速度更快，占用资源更少。

> 假如宿主机是macos，Docker实际上会通过启动一个轻量级的Linux虚拟机（如Docker Desktop中的虚拟机）来提供Linux内核支持。因此，即便是你在macOS上运行Docker，容器最终还是依赖这个虚拟机的Linux内核。

#### :airplane:docker（macos）

- docker的虚拟化层，因为macos和centos的内核不一样，docker不能直接让centos容器跑在macos上，为了解决这个问题，docker会在你的macos上先启动一个**轻量级的虚拟机**（docker desktop自带的），这个虚拟机会运行Linux内核

  - 目前没有Docker镜像可以直接使用macOS的内核。Docker容器本质上是基于**Linux内核的技术**，它依赖Linux的特性（如cgroups和namespaces）来实现资源隔离和管理。因此，所有的Docker容器默认都需要运行在Linux内核之上。
  - 无论你在Docker里拉取多少个不同的镜像（比如CentOS、Ubuntu等），这些容器实际上都**共享同一个Linux虚拟机内核**。
  - **容器的独立性**：虽然它们共享内核，但每个容器内部的文件系统、进程、网络等都是相互隔离的。你可以在一个容器里运行CentOS，在另一个容器里运行Ubuntu，它们看起来像是独立的系统，实际上都在同一个Linux内核上运行。

- 共享Linux内核：centos容器并不是运行在macos上的，而是运行在这个虚拟机里的Linux内核上。Docker会把你的CentOS镜像中的所有文件和应用放进这个虚拟机里运行，但你自己不需要管理这个虚拟机，Docker帮你处理了这一切。

- 快速体验centos：你会感觉像是直接在macOS上运行了一个CentOS环境，但实际上，它是通过虚拟机的Linux内核实现的。

  记住：macOS和Docker容器之间并没有直接的内核共享，但Docker让这一切看起来很透明，你操作起来感觉就像是在macOS上直接运行CentOS一样。

2，启动速度：

- **虚拟机**：启动一个虚拟机跟开一台物理机差不多，可能需要几十秒甚至几分钟，因为它要启动整个操作系统。
- **Docker**：Docker容器启动非常快，几秒钟内就能搞定，因为它不需要完整的操作系统，只要启动应用就行。

3，资源占用

- **虚拟机**：每个虚拟机都需要单独分配CPU、内存、存储等，资源占用比较大。
- **Docker**：Docker容器共享宿主机的资源，资源利用更高效，一台机器上可以跑很多个Docker容器，而虚拟机就不一定能做到这么多。

4，隔离性

- **虚拟机**：虚拟机之间是完全隔离的，因为每个虚拟机都有自己的操作系统和内核，安全性和隔离性很好。
- **Docker**：Docker容器是共享宿主机内核的，隔离性比虚拟机稍弱一些，但在大多数应用场景下是足够的。