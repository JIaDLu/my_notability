## 基础概念

#### :dart:创建Vue3工程

基于 vue-cli 创建

```bash
sudo npm install -g @vue/cli

vue --version

vue create project_name

cd project_name

npm install

npm run serve
```

基于 vite 创建：轻量快速

```bash
npm create vue@latest
```

![image-20240725104017731](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202407251040756.png)

```bash
cd hell_vue3

npm i

npm run dev
```

 解释重点文件：

- `env.d.ts`: ts代码不认识.jpg .txt等这些文件，这个文件的作用就是让ts代码去认识这些文件

- `index.html`: 整个Vue3项目的入口文件（假设没有引入src/main.ts里面的东西，只会展示index.html的内容）

- `package.json`and`package-lock.json`：依赖包的声明文件 

- vite.config.ts: 整个工程的配置文件，能够安装插件，配置代理

- **src**:
  - main.ts -------------  createApp(App).mount( '#app' )  解释: createApp在创建应用，每个应用得有一个根组件(App), 创建完了这个成果摆在(挂载)在id为app的div里面{这个div在index.html里面}   所以，在index.html里面，你必须得写摆‘花盆’的位置<div> 另一个必须在<script>引入main.ts的东西。
  
  - App.vue ----------- App.vue是vue应用程序的根组件。`App.vue` 文件的结构和其他 Vue 单文件组件类似，包含 `template`、`script` 和 `style` 部分 **这个是整个应用的主组件**
  
    ![image-20240725112744448](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202407251127474.png)
  
  - components 
  - assets

#### :satellite:选项式API与组合式API

#### :racehorse:响应式数据

#### :lantern:计算属性computed

computed是一种特殊的属性(方法)，用于声明一个基于其他数据的计算结果。这些结果会被缓存起来，只有当其所依赖的数据发生变化，计算结果还会重新计算。

> 换句话说，是有其所依赖的数据没有发生变化，它就不会不停地调用。对比一个function，它是没有缓存的，用其计算则会用一次调用一次

使用computed的优势：

- **性能优化**：computed属性会缓存计算结果，只在依赖的数据变化时才重新计算，而普通函数每次调用都会重新计算。
- **代码简洁**：使用computed可以让你的模板代码更清晰、更简洁，不需要在模板中直接进行复杂的计算。
- **自动依赖追踪**：computed属性会自动追踪依赖的数据变化，不需要手动处理依赖关系。

```vue
let fullName = computed(()=>{ return firstName.value + '-' + 'lastName.value' })
```

## 项目