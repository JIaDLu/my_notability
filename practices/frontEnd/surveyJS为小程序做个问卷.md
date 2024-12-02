## [为小程序做个问卷]如何将surveyJS应用到uniapp小程序的开发（**全程用vue3，不涉及纯html）

### 本篇不讲surveyJS怎么用，自己去官网上玩一玩就会了。。。（有问题可mail）

### 浅谈将surveyJS用在小程序上的一些“ 坑！”

### `1、直接尝试在uniapp中直接安装 surveyJS的依赖包`   ==这一步徒劳... 你可选择跳过==

### `2、通过webview的将问卷嵌入到小程序中`

> ### 为什么需要surveyJS：
>
> :sailboat:SurveyJS 是一个功能强大的 JavaScript 库，主要用处在于能够轻松创建各种类型的在线调查问卷、表单以及测验等。它提供了丰富的可定制化选项，无论是简单的信息收集表单还是复杂的、包含多种题型及逻辑判断的专业调查问卷，都可以通过它高效地构建出来。==几行代码就可以完成精美的问卷==
>
> ```vue
> <script setup lang="ts">
> import { Model } from "survey-core";
> import { json } from "../../data/survey_json.js";
> 
> const survey = new Model(json);
> </script>
> 
> <template>
>   <SurveyComponent :model="survey"></SurveyComponent>
> </template>
> ```
>
> > [!IMPORTANT]
> >
> > 题目的选项之间逻辑、
> >
> > 问卷题目之间有逻辑，
> >
> > 控制单选多选、
> >
> > 完成每个题目是否自动跳转、
> >
> > 完成所有题目是否需要提供preview页面
> >
> >  ....
> >
> > ==这些问题都会一键解决==

## 直接尝试在uniapp中直接安装 surveyJS的依赖包

### 安装官网安装：

```python
npm install survey-vue3-ui --save
```

构建时报错：

> Rollup failed to resolve import "survey-core/defaultV2.min.css" from "src/components/questionnaire/questionnaire.vue". This is most likely unintended because it can break your application at runtime. If you do want to externalize this module explicitly add it to `build.rollupOptions.external

:sailboat:解决方案：

![image-20241126152532804](/Users/jiadong/Library/Application Support/typora-user-images/image-20241126152532804.png)

### surveyJS中的标签设置不符合小程序标签

#### 构建时报错如下：

> ./components/questionnaire/questionnaire.wxss(6:33970): unexpected `, at pos 34165 
>
> 小程序端 style 暂不支持 p 标签选择器，推荐使用 class 选择器

:leaves:解决方案：==借助工具自动处理不兼容的样式==

使用 `PostCSS` 是处理 CSS 的工具，可以自动将标签选择器转换为小程序支持的 `class`。

安装命令：（sudo su）

我执行了 ：

1, 执行下面的命令：

> npm install postcss postcss-selector-replace --save-dev --legacy-peer-deps

2，创建 `postcss.config.js` 配置文件:在项目的根目录下创建 `postcss.config.js`，这是 PostCSS 的配置文件，用来定义需要使用的插件及其规则。

```js
module.exports = {
    plugins: {
        "postcss-selector-replace": {
            rules: [
                { search: /^p$/, replace: ".survey-p" }, // 替换 `p` 为 `.survey-p`
                { search: /^span$/, replace: ".survey-span" } // 替换 `span` 为 `.survey-span`
            ]
        }
    }
};
```

3，调整 CSS 文件引入路径

PostCSS 会处理你通过 `import` 或 `@import` 引入的 CSS 文件，因此需要确保样式文件路径正确。

```js
import "survey-core/defaultV2.min.css"; // 目标文件路径
```

**构建自动处理** 启动构建时，PostCSS 会自动修改样式，确保符合小程序规范。

##### 处理bug：

==BUT:==  <u>构建后发现postcss似乎没有发挥作用，我需要在本地测试一下</u>。

我在项目的根目录下创建了一个简单的 CSS 文件（`test.css`）：

```css
p {
    color: red;
}
```

###### 在根目录下，使用终端执行：npx postcss test.css -o output.css

你可能会遇到这个报错：

:racing_car:npm error could not determine executable to run npm error A complete log of this run can be found in: /.npm/_logs/2024-11-26T07_07_15_525Z-debug-0.log

出现这个错误通常是因为你的 Node.js 或 npm 环境配置存在问题，导致无法正确运行 `npx postcss` 命令

==因为我的 npm 环境问题较多，所以我尝试用 Yarn 替代 npm==

###### yarn add postcss-cli --dev

安装后，问题解决，不过又遇到新的问题：

:satellite:Error: Loading PostCSS Plugin failed: Be sure to have "before" and "after" object names

​	如果你在配置中使用了自定义插件，需要明确 `before` 和 `after` 的逻辑，确保插件正确运行。例如：

###### 解决方案：修改`process.config.js`文件：

```python
module.exports = {
    plugins: [
        {
            postcssPlugin: "custom-selector-replace",
            Once(root) {
                root.walkRules((rule) => {
                    if (rule.selector === "p") {
                        rule.selector = ".survey-p"; // 替换 p 标签为类选择器
                    }
                    if (rule.selector === "span") {
                        rule.selector = ".survey-span"; // 替换 span 标签为类选择器
                    }
                });
            }
        }
    ]
};
```

再次执行：npx postcss ./test.css -o ./output.css 成功！

![image-20241126154500665](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411261545775.png)

还是不行。。。

反正就是surveyJS兼容不了uniAPP，更何况gpt都这么说了...

![image-20241128224126518](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411282337949.png)

## 通过webview的将问卷嵌入到小程序中[webview也用vue3写]

### vite脚手架创建了一个新的项目文件

快速搞了一些基本配置（vite.config,ts, router/index.ts,  清除没用组件等等）

参考surveyJS官网的案例很快把问卷引了进来，并实现了一些业务上的样式需求

![image-20241128234655880](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411282346941.png)

### 打包build

要build到服务器上，这里也弄了一些deploy的配置

deploy/deploy-test.mjs文件，配置一些server信息以及上传服务器的目标目录

在package.json中的scripts中写了构建的脚本指令

![image-20241128235733700](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411282357760.png)

用nginx配置了一个url目录，这是一个项目下的子项目

### 成功在网站上显示之后，就可以通过webview引到uniapp中

#### [小程序仅支持加载网络网页，不支持本地html]

<u>uniapp提供的示例纯html  但我是用vue3写的，所以还不太能直接无脑用...</u>

根据html示例 https://gitcode.net/dcloud/uni-app/-/raw/dev/dist/uni.webview.1.5.6.js

下载了`uni.webview.1.5.6.js`的文件到本地项目中，然后直接引进来，使了一下，可以

```js
<script setup lang="ts">
import { Model } from "survey-core";
import { onMounted, onUnmounted } from "vue";
import { useRoute } from 'vue-router';
import { json } from "../data/survey_json.js"; 
import surveyInterface from '@/api/survey';
import "@/data/uni.webview.1.5.6.js"  // https://gitcode.net/dcloud/uni-app/-/raw/dev/dist/uni.webview.1.5.6.js 下载到本地

const initializeUniBridge = () => {
  const handleBridgeReady = () => {
    console.log("UniAppJSBridge 已就绪");

    // 获取当前环境
    uni.getEnv((res) => {
      console.log("当前环境：", JSON.stringify(res));
    });
  };
};
</script>
```

uniapp-webview.js这个文件并没有显式的定义对象... 

![image-20241129091853067](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411290918187.png)

但是这个uni.webview.1.5.6.js并没有显式定义uni对象...

![image-20241129091213998](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411290912083.png)

在build到服务器的时候 这个uni.webview.1.5.6.js就是弄不上去，

线上版一直会报错，`'uni' is not found...`，但本地是ok的...

:face_with_thermometer:以为是uni.webview.1.5.6.js位置不对，换了很多文件夹。后面找了很久，发现这个文件是有被编译的，只不过在编译的时候被mess up 混淆了，里面的方法在外部调的时候 找不到...

==解决方法：==

将`uni.webview.1.5.6.js`放在服务器上，然后直接引服务器上的链接

```js
onMounted( () => {
    const recaptchaScript =  document.createElement("script");
    recaptchaScript.setAttribute(
        "src",
        "https://xxx.com/survey/uni-webview.js"  // uni.webview.1.5.6.js 改名 uni-webview.js
      );
    recaptchaScript.setAttribute(
        "type",
        "text/javascript"
      );
    recaptchaScript.async = true;
    document.head.appendChild(recaptchaScript);
});
```

:yum:这会在线上就可以了... 可以调用uni对象... 我在代码中加了 点击完成问卷之后`uni.postmessage()`方法

```python
// 添加问卷完成事件
  survey.onComplete.add(alertResults);
```

那么这会就去uniapp加上这个线上版的问卷链接...

![image-20241129093200224](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411290932332.png)

页面正常显示，点击完成问卷之后，没有任何反应...

但能保证下面这个是通的...

```python
  uni.getEnv((res:any) => {
      alert("当前环境：" + JSON.stringify(res));
  });
```

通过用alert 测了一下：

发现在小程序的时候弹出==居然还是 H5==

![image-20241129093708255](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411290937378.png)

还是去看uniapp官方，结果发现一个愚蠢错误。。。

![image-20241129093827480](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411290938523.png)

其中，在微信的miniprogram 也有个外部链接，同样把它放在服务器上，再引进来...

==最终的onmounted()代码如下：==

```js
onMounted( () => {
  var userAgent = navigator.userAgent;
  if (userAgent.indexOf('AlipayClient') > -1) {
    document.writeln('<script src="https://appx/web-view.min.js"' + '>' + '<' + '/' + 'script>');
  } else if (/QQ/i.test(userAgent) && /miniProgram/i.test(userAgent)) {
    document.write(
      '<script type="text/javascript" src="https://qqq.gtimg.cn/miniprogram/webview_jssdk/qqjssdk-1.0.0.js"><\/script>'
    );
  } else if (/miniProgram/i.test(userAgent) && /micromessenger/i.test(userAgent)) {
    // 微信小程序 JS-SDK 如果不需要兼容微信小程序，则无需引用此 JS 文件。
    alert("weixinMP")
    const wePlugin =  document.createElement("script");
    wePlugin.setAttribute(
        "src",
        "https://XXX.com/survey/jweixin-1.4.0.js"  // 换成自己的服务器
      );
      wePlugin.setAttribute(
        "type",
        "text/javascript"
      );
      wePlugin.async = true;
    document.head.appendChild(wePlugin);
  } else if (/toutiaomicroapp/i.test(userAgent)) {
    document.write(
      '<script type="text/javascript" src="https://s3.pstatp.com/toutiao/tmajssdk/jssdk-1.0.1.js"><\/script>');
  } else if (/swan/i.test(userAgent)) {
    document.write(
      '<script type="text/javascript" src="https://b.bdstatic.com/searchbox/icms/searchbox/js/swan-2.0.18.js"><\/script>'
    );
  } else if (/quickapp/i.test(userAgent)) {
    // quickapp
    document.write('<script type="text/javascript" src="https://quickapp/jssdk.webview.min.js"><\/script>');
  }
  
  const recaptchaScript =  document.createElement("script");
  recaptchaScript.setAttribute(
      "src",
      "https://XXX.com/survey/uni-webview.js"    // 换成自己的服务器
    );
  recaptchaScript.setAttribute(
      "type",
      "text/javascript"
    );
  recaptchaScript.async = true;
  document.head.appendChild(recaptchaScript);

  // 添加问卷完成事件
  survey.onComplete.add(alertResults);
});

const alertResults = (sender: any) => {
  const questionnaireResults = JSON.stringify(sender.data);
  const para = { data: {action : questionnaireResults} };

  // 使用 uni.postMessage 发送数据
  uni.getEnv((res:any) => {
      alert("当前环境：" + JSON.stringify(res));
  });
  uni.postMessage(para)
};
```

基本到这就快ok了

![image-20241129094337022](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411290943166.png)

![image-20241129094614961](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411290946118.png)

并且：uni.postMessage(para);正常发了消息：

![image-20241129094720618](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411290947694.png)

但是`@message="handleWebviewMessage"`没有正确在uniapp调用handleWebviewMessage 方法

```python
<web-view src="https://tangdou-test.9pinus.com/survey/" @message="handleWebviewMessage"></web-view>
```

![image-20241129094910806](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411290949936.png)

结果：每次测的时候在uniapp都是热加载，==单页调试==... 然后设定的情况是 ：在网页发信息时，就要让uniapp应用响应收到这个消息。

但是`@message`：网页向应用 `postMessage` 时，会在特定时机（==后退、组件销毁、分享==）触发并收到消息。

![image-20241129100020040](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411291000203.png)

修改代码，点击完成问卷后的响应代码：

```js
const alertResults = (sender: any) => {
  const questionnaireResults = JSON.stringify(sender.data);
  const para = { data: {action : questionnaireResults} };

  uni.getEnv((res:any) => {
      alert("当前环境：" + JSON.stringify(res));
  });
  uni.postMessage(para);
  uni.navigateBack({delta:1}) // 让它发完消息就返回后退，以此来触发 @message
};
```

![image-20241129100402150](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411291004281.png)

==成功！ 完结撒花～🎉==

为什么不用@onPostMessage，网页向应用实时 `postMessage`

![image-20241129100526523](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411291005579.png)

