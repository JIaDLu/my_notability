## web服务(前端)

### vue3项目的开发和构建

:hong_kong:`vite.config.ts`

```ts
export default deginConfig(({command}) => {
  return {
    plugins:[vue()],
    base:'/survey/',
    
  }
})
```

base:'/survey'

含义：设置静态资源的基础路径。生成的所有资源路径（如JS、CSS 文件）都将以/survey/为前缀。

如果你的项目部署在 `https://example.com/survey/`，静态资源路径必须以 `/survey/` 开头。

> [!NOTE]
>
> ![image-20241128145552928](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411281455979.png)
>
> 构建后的index.html文件中可以看到静态资源路径：
>
> ![image-20241128145651830](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411281456881.png)

如果项目部署在服务器根路径 `/`，`base` 应设置为 `'/'` 或省略。

```python
    server: {
      proxy: {
        '^/(tangpang|api)/.*$': {
          target: "https://meinian-tangpang.9pinus.com/",
          changeOrigin: true,  //是否跨域
          secure: true,   //是否https
        }
        
      },
      watch: {
        ignored: ["survey-creator-vue", "survey-vue3-ui"].map((m) => `!**/node_modules/${m}/**`)
      }
    }
```

proxy: 配置开发服务器的代理，将==特定路径的请求==转发到==目标服务器==

例如：前端发起`http://localhost:3000/api/user` 请求。

代理将其转发为 `https://meinian-tangpang.9pinus.com/api/user`。

:sailboat:`router/index.ts`

**createWebHistory('/survey/')** 表示==所有路由==的基础路径为 `/survey/`，即：

* path: '/'实际对应 ‘/survey/’
* 如果Vue Router 假定路由基础路径为 `/`，即项目部署在 **服务器根目录**。（这里的“根目录” 并不是指服务器文件系统的根目录 / ）

Nginx 配置中的 `location /是 Nginx 中匹配请求路径的 **URL 路径根**

它匹配的是 **请求的 URL 路径**（不包括域名部分）

```nginx
server {
    listen 80;
    server_name example.com;

    # 根目录项目
    location / {
        root /path/to/root-project/dist; 
        index index.html;
        try_files $uri /index.html;
    }

    # 子目录项目 /survey/
    location /survey/ {
        root /path/to/survey-project/dist;
        index index.html;
        try_files $uri /index.html;
    }
}
```



#### Nginx 的 `location` 与 URL 路径的关系

Nginx 配置中的 location /的 “根目录” 并不是指服务器文件系统的根目录，而是 Nginx 中匹配请求路径的 **URL 路径根**。

![image-20241128153030062](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411281530840.png)

```python 
server {
    listen 80;
    server_name example.com;

    location / {
        root /path/to/your/project/dist;
        index index.html;
        try_files $uri /index.html;
    }
}
'''
location /：
匹配 所有以 / 开头的请求  路径  。
这是一个通配规则，适用于绝大多数请求路径（如 /about、/contact 等）。
root：
指定 URL 路径对应的文件系统路径。
root /path/to/your/project/dist; 的含义是：
URL / 对应文件系统路径 /path/to/your/project/dist/。
URL /about 对应文件系统路径 /path/to/your/project/dist/about。
'''
```

#### **文件路径解析规则**

当用户访问 `https://example.com/` 时：

1. 浏览器发送请求路径 `/`。
2. Nginx 通过 `root` 指定的路径将请求映射到 `/path/to/your/project/dist/`。
3. Nginx 会尝试返回 `/path/to/your/project/dist/index.html`。

```python
server {
    listen 80;
    server_name example.com;

    location /survey/ {
        root /path/to/your/project/dist;
        index index.html;
        try_files $uri /index.html;
    }
}
'''
location /survey/：
匹配所有以 /survey/ 开头的请求路径。
例如：/survey/、/survey/about 都会匹配这个规则。
文件路径解析：
URL /survey/ 对应文件系统路径 /path/to/your/project/dist/。
URL /survey/about 对应文件系统路径 /path/to/your/project/dist/about。
'''
```

:jack_o_lantern:误解

* **错误理解**：以为 location /中的 `/` 是文件系统的根目录（如 Linux 的 `/`）。
* **实际含义**：`/` 是指匹配所有 URL 路径以 `/` 开头的请求。

> #### 如果我配置了location /根目录的项目，并且在又配置了/survey/的子项目， 当用户向浏览器发生请求时，https://example.com/survey/ 那它怎么知道是去找location /根目录的项目的survey路由组件，还是去响应/survey/的子项目的idnex.html呢？

#### **当用户访问`https://example.com/survey/时：**

1. Nginx 会首先匹配 `location /survey/，因为 /survey/ 是更长的前缀匹配。
2. 请求被映射到 /path/to/survey-project/dist/index.html。
3. location / 不会被触发，因为`/survey/ 的匹配更优先。

#### **当用户访问https://example.com/`时：**

1. Nginx 匹配 `location /。
2. 请求被映射到/path/to/root-project/dist/index.html`。

#### **冲突场景**

当用户访问 https://example.com/survey/` 时：

* Nginx 优先匹配/survey/ 的子项目配置，因为location /survey/` 优先级更高。
* 根项目的 /survey/ 组件永远不会生效，访问 /survey/` 时总是加载子项目。

#### :school_satchel:使用子域名部署子项目

如果你希望子项目保持独立性，可以为它设置一个子域名（如 `survey.example.com`）。

```nginx
server {
    listen 80;
    server_name survey.example.com;

    # 子项目
    root /path/to/survey-project/dist;
    index index.html;
    try_files $uri /index.html;
}

server {
    listen 80;
    server_name example.com;

    # 根项目
    root /path/to/root-project/dist;
    index index.html;
    try_files $uri /index.html;
}
```

* 根项目和子项目完全隔离，不会产生冲突。
* 更适合子项目独立开发和管理。

配置路由基础路径

```python
import { createRouter, createWebHistory } from 'vue-router'
import Questionnaire from '@/components/questionnaire.vue'

const router = createRouter({
  history: createWebHistory('/survey/'), // 表示所有路由的基础路径为/survey/
  routes: [
    {
      path: '/',
      name: 'home',
      component: Questionnaire,
    }
  ],
})

export default router
```

