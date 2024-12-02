











### page and router

uni.relaunch是 uni - app 框架中的一个 API（应用程序编程接口）。它用于关闭所有页面，打开到应用内的某个页面。这个操作类似于重新启动应用并跳转到指定页面，会销毁之前的页面栈，只保留新打开的页面。



### pages.json

`pages.json`是`UniApp`框架中一个非常关键的配置文件，主要用于对应用的页面路径、窗口样式、导航栏、底部`tabBar`等页面相关的全局配置进行设定。

1. **页面路径配置**
   - `pages`数组：在`pages.json`中，`pages`是一个数组，其中的每个元素代表一个应用页面的路径。例如：
   ```json
   {
       "pages":[
           {
               "path": "pages/index/index",
               "style": {
                   "navigationBarTitleText": "首页"
               }
           },
           {
               "path": "pages/detail/detail",
               "style": {
                   "navigationBarTitleText": "详情页"
               }
           }
       ]
   }
   ```
   - 这个数组的顺序决定了==应用==启动时的初始页面以及页面栈的顺序。在上述示例中，`pages/index/index`是应用启动后的第一个页面。当在`index`页面通过`uni.navigateTo`等导航方式跳转到`detail`页面时，页面栈中就会依次保存这两个页面的信息。

2. **窗口样式配置**
   - `globalStyle`对象：用于设置应用的全局窗口样式。它包含很多属性，如`navigationBarBackgroundColor`（导航栏背景颜色）、`navigationBarTextStyle`（导航栏文字颜色）等。例如：
   ```json
   {
       "globalStyle": {
           "navigationBarBackgroundColor": "#F8F8F8",
           "navigationBarTextStyle": "black",
           "backgroundColor": "#FFFFFF"
       }
   }
   ```
   - 在这里，导航栏背景色被设置为浅灰色（`#F8F8F8`），导航栏文字颜色为黑色（`black`），页面背景色为白色（`#FFFFFF`）。这些设置会应用到应用中的所有页面，除非在具体页面的`style`配置中进行了覆盖。

3. **导航栏配置**
   - 在每个页面的`style`对象中，可以对导航栏进行详细配置。除了前面提到的`navigationBarTitleText`（导航栏标题文字）属性外，还有`navigationBarHidden`（是否隐藏导航栏）等属性。例如：
   ```json
   {
       "pages":[
           {
               "path": "pages/index/index",
               "style": {
                   "navigationBarTitleText": "首页",
                   "navigationBarHidden": false
               }
           }
       ]
   }
   ```
   - 上述代码表示`index`页面显示导航栏，标题为“首页”。如果将`navigationBarHidden`设置为`true`，则该页面将隐藏导航栏。

4. **底部tabBar配置**
   - `tabBar`对象：用于配置应用底部的`tabBar`。它包含`color`（未选中图标和文字颜色）、`selectedColor`（选中图标和文字颜色）、`list`（`tabBar`列表项）等属性。例如：
   ```json
   {
       "tabBar": {
           "color": "#999999",
           "selectedColor": "#333333",
           "list": [
               {
                   "pagePath": "pages/index/index",
                   "text": "首页",
                   "iconPath": "static/tabbar/home.png",
                   "selectedIconPath": "static/tabbar/home-selected.png"
               },
               {
                   "pagePath": "pages/mine/mine",
                   "text": "我的",
                   "iconPath": "static/tabbar/mine.png",
                   "selectedIconPath": "static/tabbar/mine-selected.png"
               }
           ]
       }
   }
   ```
   - 这个配置定义了一个包含两个选项的`tabBar`。未选中时，图标和文字颜色为灰色（`#999999`），选中时为深灰色（`#333333`）。每个`tab`项都指定了对应的页面路径、显示文字、图标路径和选中图标路径。

### manifest.json

 1. **概述**

   - 在UniApp项目中，`manifest.json`文件是非常重要的配置文件。它用于配置==应用==的基本信息、权限设置、第三方SDK集成等众多关键内容。这个文件以JSON（JavaScript Object Notation）格式编写，使得数据结构清晰，易于理解和编辑。
2. **主要配置项**
   - **应用基本信息**
     - **appid**：这是应用的唯一标识符，用于在应用商店等平台区分不同的应用。例如，在发布到App Store或安卓应用市场时，这个id是必不可少的。
     - **name**：应用的名称，会显示在设备的应用列表中。比如你的应用叫“我的购物助手”，这个名称就会出现在用户的手机桌面上。
     - **description**：应用的描述信息，用于向用户简要介绍应用的功能和用途。在应用市场中，用户查看应用详情时可以看到这个描述，它有助于吸引用户下载和使用应用。
   - **应用启动配置**
     - **entryPagePath**：指定应用启动时加载的页面路径。例如，`"entryPagePath": "pages/index/index"`表示应用启动时会先加载`pages/index/index`这个页面，它就像网站的首页一样，是用户打开应用首先看到的内容。
   - **页面路径配置**
     - **pages**：这是一个数组，用于列出应用中所有的页面路径。每个元素代表一个页面，格式通常为`{ "path": "pages/page1/page1", "style": { "navigationBarTitleText": "页面1标题" } }`。其中`path`指定页面的实际路径，`style`可以用来配置页面的导航栏标题等样式信息。
   - **权限配置**
     - **permissions**：用于声明应用需要的权限。例如，如果你的应用需要访问用户的摄像头，就需要在这里添加摄像头权限相关的配置。像`{ "name": "camera" }`这样的配置项表示应用需要使用摄像头权限，当用户安装应用时，系统会根据这个配置向用户请求相应的权限。
   - **第三方SDK配置（如推送服务等）**
     - **uni_modules**：如果你的应用集成了第三方的UniApp模块，或者使用了一些扩展插件（如推送插件、地图插件等），可以在这里进行配置。以推送服务为例，可能需要配置推送服务提供商的相关参数，如`appkey`、`appsecret`等，用于初始化推送服务，使得应用能够接收推送消息。
3. **平台相关配置**
   - `manifest.json`还支持针对不同平台（如iOS、Android、H5等）进行差异化配置。
   - **iOS配置**
     - 可以配置应用的图标尺寸、启动图尺寸等符合iOS规范的内容。例如，`"ios": { "appIcon": [ { "size": "20x20", "path": "static/icon/ios/icon-20.png" } ], "launchImage": [ { "width": "750", "height": "1334", "path": "static/launch/ios/launch-750x1334.png" } ] }`，这里配置了iOS应用图标和启动图的具体尺寸和路径。
   - **Android配置**
     - 同样可以配置图标、启动图等，并且还可以配置一些Android特有的参数，如`"android": { "permissions": [ "android.permission.WRITE_EXTERNAL_STORAGE" ], "appIcon": [ { "size": "mipmap - hdpi", "path": "static/icon/android/hdpi.png" } ], "launchImage": [ { "width": "720", "height": "1280", "path": "static/launch/android/launch-720x1280.png" } ] }`，其中`permissions`配置了应用在Android平台需要的额外权限，如存储权限。
4. **与打包发布的关系**
   - 在将UniApp项目打包成不同平台的应用时，打包工具会读取`manifest.json`中的配置信息。这些信息会被用于生成符合对应平台规范的安装包。例如，根据配置的应用名称、图标、权限等内容，打包出能够在App Store或安卓应用市场上架的应用安装包。