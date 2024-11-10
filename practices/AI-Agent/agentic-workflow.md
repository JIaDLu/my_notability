## Coze构建慢病管理的智能体工作流

:sailboat:补充异步知识储备

假如我们需要从多个远程API中获取不同用户的数据。传统同步方式会一个一个地等待请求完成，而异步方式可以让所有请求并发地执行。异步函数允许“非阻塞”地等待一个操作完成，避免程序在等待时停滞不前，从而提高执行效率。

1. 概念：**异步函数**由 `async def` 关键字定义，通常与 `await` 一起使用。它允许在一个任务中暂停执行，让其他任务继续运行，从而实现高效的非阻塞并发。

2. 术语：

   * 事件（event）：表示需要等待或处理的操作，例如网路请求、文件IO等。异步编程中，当任务遇到`await`时, 这个操作会成为“事件”，需要等待完成。
   * 事件循环（event Loop）：负责管理和调度所有的异步任务。当一个任务需要等待时，事件循环会将控制权交给其他任务，避免阻塞整个程序的运行。
   * **`async` 和 `await`**：`async` 定义异步函数，`await` 用于在异步操作上暂停任务并将控制权交回给事件循环。

3. 单线程并发

   异步编程中的单线程并发通过事件循环和非阻塞操作，使单个线程能够高效地管理多个任务，不同于多线程并行的机制。

4. 异步函数通过 `asyncio` 和 `await` 可以并发地运行多个任务，如模拟文件下载或网络数据获取等。在等待某个任务完成的同时，其他任务可以继续执行，提升程序的响应性和效率。

```python
import asyncio
import random

async def download_file(file_id):
    print(f"开始下载文件 {file_id}...")
    # 模拟下载时间（随机等待1-3秒）
    await asyncio.sleep(random.randint(1, 3))
    print(f"文件 {file_id} 下载完成！")

async def main():
    tasks = [download_file(i) for i in range(1, 6)]  # 创建5个下载任务
    await asyncio.gather(*tasks)  # 并发地运行所有任务

# 启动事件循环并运行异步任务
asyncio.run(main())
```

当一个任务在 `await asyncio.sleep()`（或其他异步操作）时让出控制权后，**当这个“事件”完成后，事件循环会重新调度这个任务继续执行**。具体来说：

* 当 `await asyncio.sleep()` 的等待时间结束时，该任务会进入“就绪”状态，意味着它已经准备好继续执行后续代码。

* 事件循环会在合适的时机“唤醒”这个任务并让它恢复执行。任务的控制权从事件循环中重新获取，接着执行 `await` 之后的代码。







### API接口调用更全面的数据：根据用户询问的内容，调用API获取更多患者数据 提供给AI，让AI做出更加精确的回答。

将API接口做成一个插件，在工作流里调用获取更多输出数据提供给AI

![image-20240909201407045](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411091834901.png)

![image-20240909201221095](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411091834436.png)

![image-20240909201334302](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411091835389.png)



#### 测试：

{"query":"最近血糖怎么样呀？有没有好转点呢","config":{"tangdou_patientId":"28226","tangdou_orderId":"311","称呼":"家栋","tangdou_apiHost":"https://maiya.9pinus.com/tangpang"}}

{"query":"什么时候可以吃麻辣烫？","config":{"tangdou_patientId":"28226","tangdou_orderId":"311","称呼":"家栋","tangdou_apiHost":"https://maiya.9pinus.com/tangpang"}}



![image-20240910210432297](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411091835851.png)

![image-20240910210720803](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411091835277.png)

### 创建知识库：基于知识库给用户做出更加精准的回答



![image-20240911175801336](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411091832697.png)

![image-20240911175822022](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411091834171.png)

记录器API

![image-20240914225926733](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg/img/202411091834152.png)