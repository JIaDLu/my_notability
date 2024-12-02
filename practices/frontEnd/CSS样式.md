## CSS样式

### 基本语法

1. 选择器

   选择器用于选定html中的元素，css样式会应用到这些元素上。

   * 元素选择器：直接选择标签名

   * 类选择器：以 `.` 开头，选择有相同类名的元素。

     ```css
     .box {
       background-color: yellow;
     }
     ```

   * 组合选择器：比如选择某个类中的某个标签

     ```css
     .content p {
       color: green;
     }
     ```

2. 属性与属性值

   color：

   ```css
   h1 {
     color: red;
   }
   ```

   background-color：

   ```css
   div {
     background-color: lightgray;
   }
   ```

   font-size：

   ```css
   p {
     font-size: 16px;
   }
   ```

   width,height

   ```python
   img {
     width: 100px;
     height: auto;
   }
   ```

   `margin`：元素外的间距

   `padding`：元素内的间距

   ```python
   .box {
     margin: 10px;
     padding: 20px;
   }
   ```

3. 盒模型

   盒模型一个元素占据的空间，包括：

   * **内容（content）**：元素的实际内容。

   * **内边距（padding）**：内容和边框之间的空白。

   * **边框（border）**：元素的边框。

   * **外边距（margin）**：元素与外部其他元素的间距。

4. 布局

   flexbox布局：用于横向或纵向布局

   ```css
   .container {
     display: flex;
     justify-content: space-between;
   }
   ```

5. 字体和文本样式

   * **字体（font-family）**：设置字体。
   * 字体加粗（font-weight）
   * 文本对齐（text-align）

6. 伪类和伪元素

   * **伪类**：用于给元素的某种状态添加样式（如鼠标悬停）。

     ```css
     a:hover {
       color: red;
     }
     ```

   * **伪元素**：用于选择元素的一部分（如首字母）。

     ```css
     p::first-letter {
       font-size: 2em;
       color: green;
     }
     ```

7. 位置

   * 默认值static：默认位置。元素按照正常的文档流排列，不进行特殊定位。
   * relative（相对定位）
   * absolute（绝对定位）
   * fixed（固定定位）

### 项目样式

从项目的角度构建vue3项目的CSS需要系统化的设计，确保样式结构清晰、可维护，并能适应项目的增长。

1. 全局与局部样式分离

   **全局样式**：将全局的通用样式（如字体、颜色、按钮样式、布局规则等）集中放在一个或几个文件中，并在项目入口（如 `main.js`）中引入。

   ```python
   // main.js
   import './assets/styles/global.css';
   ```

   **局部样式**：组件内部的样式使用 `<style scoped>`，保持样式的局部作用，避免全局样式污染组件。

2. 合理使用CSS预处理器

   使用scss预处理器，可以更高效地编写和管理CSS，常见的做法包括：

   * 使用变量管理颜色、字体、间距等，确保全局统一性。

   * 使用嵌套和继承来减少代码重复，增加代码的可读性。

     ```css
     /* 变量定义 */
     $primary-color: #3498db;
     $secondary-color: #2ecc71;
     
     .button {
       background-color: $primary-color;
       color: white;
       
       &:hover {
         background-color: $secondary-color;
       }
     }
     ```

     进而在vue项目中，可以通过`lang="scss"` 直接使用 SCSS：

     ```css
     <style lang="scss" scoped>
     @import '@/assets/styles/variables.scss';
     
     .box {
       color: $primary-color;
     }
     </style>
     ```

3. 模块化 CSS 和组件化设计

   将每个 Vue 组件的 CSS 限制在组件内部，使用 `scoped` 或 CSS Modules 来防止样式互相影响。这样，你可以让每个组件的样式独立、可维护，并且不会破坏全局样式。

   ```python
   <template>
     <div class="profile-card">
       <h2 class="profile-card__title">Profile</h2>
     </div>
   </template>
   
   <style scoped>
   .profile-card {
     border: 1px solid #ccc;
     padding: 10px;
   }
   
   .profile-card__title {
     color: #333;
   }
   </style>
   
   ```

4. 主题化与样式切换

   通过创建主题文件和样式变量，你可以轻松实现项目的样式切换。比如，可以创建不同的 SCSS 变量文件来定义不同的主题色。

   ```scss
   /* variables-dark.scss */
   $primary-color: #121212;
   $secondary-color: #1e1e1e;
   $font-color: #ffffff;
   ```

   ```scss
   /* variables-light.scss */
   $primary-color: #ffffff;
   $secondary-color: #f0f0f0;
   $font-color: #000000;
   ```

   通过控制加载不同的变量文件，可以实现项目主题的动态切换。

5. 项目结构：

   **按模块分文件夹**：比如 `assets/styles/` 中有全局样式，按页面模块划分的样式文件放在 `components/` 文件夹下。

   **建立基础样式文件**：如 `variables.scss`、`mixins.scss`，放置常用的样式工具、颜色、字体等。

   ```python
   src/
   │
   ├── assets/
   │   └── styles/
   │       ├── variables.scss  // 全局变量
   │       ├── global.scss     // 全局样式
   │       └── mixins.scss     // CSS Mixins
   │
   ├── components/
   │   ├── Header.vue          // 头部组件
   │   ├── Footer.vue          // 底部组件
   │   └── ProfileCard.vue     // 用户资料卡组件
   │
   └── views/
       ├── Home.vue            // 首页
       └── About.vue           // 关于页面
   ```

   