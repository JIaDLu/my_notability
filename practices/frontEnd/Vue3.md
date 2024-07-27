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

ref()  reactive()



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

#### :cactus:自定义组件

##### `props` 是组件之间传递数据的机制。

父组件通过`props`向子组件传递数据，子组件通过声明`props`来接收这些数据。

> 如何使用props 

1. 在子组件中声明它期望接收的props

   ```vue
   <script>
   props = {
       taskPackageId: String,
       nodeData: Object,
       elements: Array
   }
   // 在这个例子中，子组件声明了三个props：taskPackageId、nodeData和elements，分别期望接收一个字符串、一个对象和一个数组。
   </script>
   ```

2. 在父组件中传递props: 通过在子组件🉐️标签上使用`:`语法来传递`props`：

   ```vue
   <script>
   <actionBar 
     :taskPackageId="pageData.taskPackageId" 
     :nodeData="selectNodePropData" 
     :elements="elements">
   </actionBar>
   //这里，父组件传递了三个props：
   //  taskPackageId绑定到pageData.taskPackageId
   //  nodeData绑定到selectNodePropData
   //  elements绑定到父组件中的elements变量
   </script>
   ```

通过`props`的使用，父组件可以将数据传递给子组件，使子组件可以根据这些数据进行渲染和逻辑处理。

##### 插槽

- 默认插槽允许在子组件的特定位置插入父组件提供的内容，从而实现组件之间的灵活组合和内容分发

> 使用：

1. 在子组件(child-component.vue)中定义插槽：

   ```vue
   <template>
     <div class="child-component">
       <slot></slot>
     </div>
   </template>
   ```

2. 在父组件中使用插槽：

   ```vue
   <template>
     <child-component>
       <p>This is passed to the slot</p>
     </child-component>
   </template>
   ```

- 具名插槽：定义多个插槽并为它们命名

> 使用：

1. 在子组件中定义具名插槽：

   ```vue
   <template>
     <div class="child-component">
       <slot name="header"></slot>
       <slot></slot>
       <slot name="footer"></slot>
     </div>
   </template>
   ```

2. 在父组件中使用插槽：(v-shot可以简化为# )

   ```vue
   <template>
     <child-component>
       <template v-slot:header>
         <h1>Header Content</h1>
       </template>
       <p>Main Content</p>
       <template #footer>
         <p>Footer Content</p>
       </template>
     </child-component>
   </template>
   ```

#### :oden:丰富的UI组件

- <el-drawer> 抽屉组件，用于创建一个从屏幕边缘滑出的面板。



## 项目