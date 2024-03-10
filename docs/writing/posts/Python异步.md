---
date: 2024-03-09
readtime: 15
authors:
- Dewey
categories:
- Python
---

# Python Async

## Parallelism V.S. Threading

进程（Process）和线程（Thread）是计算机科学中重要的概念，它们用于管理计算机执行任务的方式。它们之间的主要区别在于它们管理资源的方式和执行任务的方式。

1. **进程（Process）**
    - 进程是操作系统分配资源的基本单位。它包含了程序代码、数据以及程序执行时所需的各种系统资源（如内存空间、文件句柄等）。
    - 每个进程都有自己独立的内存空间，一个进程不能直接访问另一个进程的内存。
    - 进程之间的切换开销相对较高，因为切换进程需要保存和恢复整个进程的状态。
    - 进程之间通常是相互独立的，它们通过进程间通信（IPC）来进行数据交换。
2. **线程（Thread）**
    - 线程是进程内部的执行单元。一个进程可以包含多个线程，它们共享进程的资源（如内存空间、文件句柄等）。
    - 线程之间共享相同的内存空间，因此可以直接访问同一进程内的数据。
    - 线程之间切换的开销较低，因为线程共享同一进程的地址空间，切换时只需保存和恢复少量的线程状态。
    - 由于线程共享同一进程的资源，线程之间的同步和通信更加容易和高效。

<!-- more -->

主要区别总结如下：

- 进程是资源分配的基本单位，而线程是操作系统调度的基本单位。
- 进程之间相互独立，而线程共享同一进程的资源。
- 进程切换开销大，线程切换开销小。
- 进程通信需要较高的开销，而线程通信相对简单。

但是在CPython解释器中由于GIL的存在，Python无法实现并行的多线程，同一时刻只有一个线程执行 Python 字节码，即并发的多线程。

我们以`Concurrent.future`中的`ProcessPoolExecutor`和`ThreadPoolExecutor`来观察进程池和线程池是如何管理和执行进程和线程的。

### ProcessPoolExecutor

对于`ProcessPoolExecutor`来说，它能够实现真正的并行处理，因为它使用的是进程而不是线程来执行任务。在Python中，每个进程都有自己独立的Python解释器和内存空间，因此不受全局解释锁（GIL）的限制，多个进程可以同时执行Python字节码，从而实现真正的并行处理。

**进程池的创建**：

```python
class ProcessPoolExecutor(_base.Executor):
    def __init__(self, max_workers=None):
        ...
        if max_workers is None:
            self._max_workers = os.cpu_count() or 1  # 默认使用CPU核心数
        else:
            if max_workers <= 0:
                raise ValueError("max_workers must be greater than 0")
            self._max_workers = max_workers
        ...
        self._processes = set()  # 存放进程的集合

```

在初始化时，`ProcessPoolExecutor`会创建一定数量的进程，并将它们存放在进程池中。默认情况下，进程池会使用系统中的CPU核心数作为最大工作进程数。

**任务分配和执行**：

```python
class ProcessPoolExecutor(_base.Executor):
    ...
    def submit(self, fn, *args, **kwargs):
        ...
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')
            f = _base.Future()
            self._pending_work_items.put((f, fn, args, kwargs))  # 将任务放入待处理队列
            self._adjust_process_count()  # 调整进程数量
        return f

```

任务提交到`ProcessPoolExecutor`时，会被放入待处理队列`_pending_work_items`中。然后，`_adjust_process_count`方法会根据需要动态地调整进程数量，确保有足够的进程来处理任务。

**进程池的管理**：

```python
class ProcessPoolExecutor(_base.Executor):
    ...
    def _adjust_process_count(self):
        ...
        while len(self._processes) < self._max_workers:  # 如果当前进程数小于最大进程数
            p = self._context.Process(target=_process_worker, args=(self._call_queue, result_queue))  # 创建新的进程
            self._processes.add(p)  # 将新进程加入进程集合
            p.start()  # 启动新进程

```

`_adjust_process_count`方法会根据当前待处理任务的数量和最大工作进程数来调整进程的数量。如果当前进程数小于最大进程数，则会创建新的进程并加入进程池中。

通过上述机制，`ProcessPoolExecutor`能够利用多个进程实现真正的并行处理。每个进程都可以独立执行任务，不受全局解释锁的限制，因此多个任务可以在同一时间段内同时执行，实现真正的并行处理。

**任务提交**：

```python
class ProcessPoolExecutor(_base.Executor):
    ...
    def submit(self, fn, *args, **kwargs):
        ...
        f = _base.Future()
        self._pending_work_items.put((f, fn, args, kwargs))  # 将任务放入待处理队列
        self._adjust_process_count()  # 调整进程数量，确保有足够的进程来处理任务
        return f

```

任务通过`submit`方法提交到进程池中，它将任务封装为一个元组，然后放入待处理队列`_pending_work_items`中。同时，`_adjust_process_count`方法会根据需要动态地调整进程数量，确保有足够的进程来处理任务。

**进程调度和执行**：

```python
class ProcessPoolExecutor(_base.Executor):
    ...
    def _adjust_process_count(self):
        ...
        while len(self._processes) < self._max_workers:  # 如果当前进程数小于最大进程数
            p = self._context.Process(target=_process_worker, args=(self._call_queue, result_queue))  # 创建新的进程
            self._processes.add(p)  # 将新进程加入进程集合
            p.start()  # 启动新进程

```

在`_adjust_process_count`方法中，会根据当前待处理任务的数量和最大工作进程数来调整进程的数量。如果当前进程数小于最大进程数，则会创建新的进程并加入进程池中，并启动这些新的进程。

**任务执行**：

```python
def _process_worker(call_queue, result_queue):
    while True:
        try:
            future, fn, args, kwargs = call_queue.get(block=True)  # 从任务队列中获取任务
        except queue.Empty:  # 如果队列为空
            break  # 退出循环
        try:
            result = fn(*args, **kwargs)  # 执行任务的可调用对象
        except BaseException as exc:
            result_queue.put((future, exc))  # 任务执行出现异常，将异常信息放入结果队列
        else:
            result_queue.put((future, result))  # 将任务的结果放入结果队列

```

在每个工作进程中，会循环地从任务队列`call_queue`中获取任务并执行。如果任务执行成功，则将任务的结果放入结果队列`result_queue`中；如果任务执行出现异常，则将异常信息放入结果队列。这样，多个工作进程可以并行地执行多个任务。

让我们来分析一下为什么这样的循环能够实现并行处理：

1. **多进程并行**：虽然每个工作进程内部的循环是顺序执行的，但是由于存在多个工作进程，这些进程可以同时运行，并且各自独立地处理任务。每个工作进程都可以并行地从任务队列中获取任务并执行，而不会受到其他进程的影响。
2. **任务调度**：任务队列`call_queue`中存放着待执行的任务，多个工作进程可以从这个队列中并发地获取任务。当有任务可用时，多个工作进程可以同时竞争获取任务并执行，从而实现任务的并行处理。
3. **结果收集**：工作进程执行完任务后，会将执行结果放入结果队列`result_queue`中。主程序可以从结果队列中获取各个工作进程执行任务的结果，从而实现结果的收集。这种方式保证了在并行处理的同时，能够正确地收集和处理任务的结果。

**结果获取**：

```python
class _base.Future:
    ...
    def result(self, timeout=None):
        ...
        if self._state == _FINISHED:
            return self._result  # 返回任务的执行结果
        ...

```

通过`Future.result`方法可以获取任务的执行结果。如果任务已经执行完毕，则直接返回任务的执行结果；如果任务还未执行完毕，则根据需要等待一定时间直至任务执行完毕。

### ThreadPoolExecutor

**线程的创建和维护**：

```python
class ThreadPoolExecutor(_base.Executor):
    def __init__(self, max_workers=None, thread_name_prefix=''):
        ...
        if max_workers is None:
            self._max_workers = min(32, (os.cpu_count() or 1) + 4)
        else:
            if max_workers <= 0:
                raise ValueError("max_workers must be greater than 0")
            self._max_workers = max_workers
        ...
        self._work_queue = queue.SimpleQueue()  # 任务队列
        self._threads = set()  # 存放线程的集合
        self._thread_name_prefix = thread_name_prefix
        self._shutdown = False

```

在初始化`ThreadPoolExecutor`时，它会创建一个任务队列`_work_queue`，用于存放提交到线程池的任务。同时，它还会维护一个线程集合`_threads`，用于存放线程池中的线程。`_max_workers`表示线程池中允许的最大线程数。

**任务队列管理**：

```python
class ThreadPoolExecutor(_base.Executor):
    ...
    def submit(self, fn, *args, **kwargs):
        ...
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')
            f = _base.Future()
            w = _WorkItem(f, fn, args, kwargs)  # 封装任务
            self._work_queue.put(w)  # 将任务放入队列
            self._adjust_thread_count()  # 调整线程数量
        return f

```

在`submit`方法中，当有任务提交到线程池时，它会封装任务为`_WorkItem`对象，并将该对象放入任务队列`_work_queue`中。同时，它会调用`_adjust_thread_count`方法来调整线程数量，确保线程池中有足够的线程来执行任务。

**线程池的状态管理**：

```python
class ThreadPoolExecutor(_base.Executor):
    ...
    def _adjust_thread_count(self):
        # 调整线程数量
        if len(self._threads) < self._max_workers:
            t = threading.Thread(target=self._worker, name=self._thread_name_prefix + str(len(self._threads)))
            t.daemon = True
            t.start()
            self._threads.add(t)

```

`_adjust_thread_count`方法用于调整线程数量，当线程池中的线程数少于最大线程数时，会创建新的线程并添加到线程池中。

**任务提交**：

```python
class ThreadPoolExecutor(_base.Executor):
    ...
    def submit(self, fn, *args, **kwargs):
        ...
        f = _base.Future()
        w = _WorkItem(f, fn, args, kwargs)  # 封装任务为_WorkItem对象
        self._work_queue.put(w)  # 将任务放入任务队列
        self._adjust_thread_count()  # 调整线程数量，确保有足够的线程来执行任务
        return f

```

任务通过`submit`方法提交到线程池中，首先封装任务为`_WorkItem`对象，然后将该对象放入任务队列`_work_queue`中。在这个过程中，还会调用`_adjust_thread_count`方法来动态调整线程数量，确保有足够的线程来执行任务。

**任务执行**：

```python
class ThreadPoolExecutor(_base.Executor):
    ...
    def _worker(self):
        ...
        try:
            while True:
                try:
                    work_item = self._work_queue.get(block=False)  # 从任务队列中获取任务
                except queue.Empty:
                    break
                ...
        finally:
            self._threads.discard(threading.current_thread())  # 从线程集合中移除当前线程

```

在**`_worker`**方法中，使用了一个**`while True`**的循环来保持线程的执行。在循环内部，线程会尝试从任务队列中获取任务执行，如果任务队列为空，则退出循环并结束线程的执行。这样，线程池中的线程可以不断地从任务队列中获取任务并执行，直到任务队列为空或者线程池被关闭。

通过这个循环，线程池能够保持线程的持续执行，实现任务的并发处理。当任务队列为空时，线程会等待新的任务被提交到任务队列中，而不是无效地等待或者占用资源。这样可以有效地利用线程池中的线程资源，提高任务的执行效率。

**任务完成**：

```python
class _WorkItem:
    ...
    def run(self):
        ...
        try:
            result = self.fn(*self.args, **self.kwargs)  # 执行任务的可调用对象
        except BaseException as exc:
            self.future.set_exception(exc)  # 任务执行出现异常，设置Future的异常状态
        else:
            self.future.set_result(result)  # 任务执行成功，设置Future的结果状态

```

任务在执行完毕后，会调用`run`方法来执行任务的可调用对象（函数）。如果任务执行成功，则通过`Future.set_result`方法设置任务的结果状态；如果任务执行出现异常，则通过`Future.set_exception`方法设置任务的异常状态。

**结果获取**：

```python
class _base.Future:
    ...
    def result(self, timeout=None):
        ...
        if self._state == _FINISHED:
            return self._result  # 返回任务的执行结果
        ...

```

通过`Future.result`方法可以获取任务的执行结果。如果任务已经执行完毕，则直接返回任务的执行结果；如果任务还未执行完毕，则根据需要等待一定时间直至任务执行完毕。

以上是`ThreadPoolExecutor`执行过程的关键步骤，通过这些机制，线程池能够高效地管理线程资源，并执行提交到线程池的任务。

## Async IO

**事件循环（Event Loop）**：

- `asyncio` 基于事件循环模型，主要的控制流程是由事件循环来管理的。
- 事件循环负责监控异步任务的状态，并在任务就绪时执行它们。
- 事件循环会持续运行，直到所有的任务都完成。

**协程（Coroutines）**：

- `asyncio` 使用协程来实现异步编程。
- 协程是一种特殊的函数，可以在其中使用 `async` 关键字定义。
- 在协程中可以使用 `await` 关键字来挂起执行，等待异步操作完成。

**异步 I/O 操作**：

- `asyncio` 支持异步 I/O 操作，包括文件 I/O、网络 I/O、进程和线程池的调度等。
- 使用异步 I/O 可以在等待 I/O 操作完成时挂起当前协程的执行，以充分利用系统资源。

**任务和协程的调度**：

- `asyncio` 提供了 `asyncio.create_task()` 函数来创建协程任务（task），并将它们提交给事件循环进行调度。
- 任务可以并发执行，并且可以相互等待、组合和取消。

**并发编程**：

- `asyncio` 可以轻松处理大量的并发任务，而不需要创建大量的线程或进程。
- 它在处理网络编程、高并发服务器、异步 I/O 操作等方面有着广泛的应用。

**异常处理**：

- `asyncio` 提供了异常处理机制，可以捕获和处理异步操作中可能发生的异常。
- 异常处理通常基于协程，可以很好地控制异常的传播。

### Event loop

当涉及到协程调度时，uvloop 主要依赖于 libuv 库提供的事件循环机制。libuv 是一个跨平台的异步 I/O 库，用于处理事件循环、套接字操作和其他底层操作。uvloop 使用 libuv 提供的事件循环来管理和调度协程。

**事件循环初始化：** uvloop 在启动时会初始化 libuv 的事件循环。这个过程通常在 uvloop 的 `_init_once()` 函数中完成。在初始化期间，uvloop 创建了一个 libuv 的事件循环对象，并将其设置为默认事件循环。

```python
def _init_once():
    uv_loop = ffi.new("uv_loop_t*")
    uv.uv_loop_init(uv_loop)
    handle = ffi.new("uv_handle_t*")
    uv.uv_unref(ffi.cast("uv_handle_t*", handle))
    handle[0] = uv_loop
    uv.uv_default_loop.argtypes = [ffi.TYPE_PTR]
    uv.uv_default_loop(handle)

```

**事件循环运行：** 一旦事件循环初始化完成，uvloop 开始运行事件循环。这个过程通常在 `_run()` 函数中完成。在事件循环运行过程中，uvloop 会持续地监听事件，并根据事件类型执行相应的操作。

```python
def _run():
    while True:
        uv.uv_run(uv.uv_default_loop(), UV_RUN_ONCE)
        _process_done_callbacks()
        _process_deferred_callbacks()

```

**协程状态管理：** uvloop 跟踪每个协程的状态，并根据需要进行调度和处理。协程的状态可以是挂起（pending）、运行中（running）、已完成（finished）等。uvloop 使用 libuv 提供的事件循环机制来检查每个协程的状态，并根据需要执行相应的操作。

```python
class Future:
    def __init__(self, loop=None):
        self._state = PENDING
        self._callbacks = []

    def set_result(self, result):
        self._result = result
        self._state = FINISHED
        self._call_callbacks()

    def add_done_callback(self, fn):
        if self._state != PENDING:
            fn(self)
        else:
            self._callbacks.append(fn)

```

**协程调度：** uvloop 使用 libuv 库中的事件循环机制来调度协程。这包括通过注册事件处理器来等待特定的 I/O 事件，然后将控制权传递给相应的协程。

```python
def _run():
    while True:
        # 调用 libuv 的事件循环函数
        uv.uv_run(uv.uv_default_loop(), UV_RUN_ONCE)
        
        # 处理已完成的协程任务
        _process_done_callbacks()
        
        # 处理延迟执行的回调任务
        _process_deferred_callbacks()
```

**事件处理：** 在事件循环运行过程中，uvloop 会监听不同类型的事件，并执行相应的操作。例如，当某个套接字可读或可写时，uvloop 会调用相应的回调函数来处理这些事件。这些回调函数通常会触发对应协程的执行。

```python
def _on_readable(handle, events, error):
    # 处理套接字可读事件
    ...

def _on_writable(handle, events, error):
    # 处理套接字可写事件
    ...

```

**What is more crucial is understanding a bit beneath the surface about the mechanics of the event loop. Here are a few points worth stressing about the event loop:**

> **#1: Coroutines don’t do much on their own until they are tied to the event loop.**
> 
> 
> You saw this point before in the explanation on generators, but it’s worth restating. If you have a main coroutine that awaits others, simply calling it in isolation has little effect:
> 
> ```python
> import asyncio
> 
> async def main():
>     print("Hello ...")
>     await asyncio.sleep(1)
>     print("World!")
> 
> routine = main()
> routine
> <coroutine object main at 0x1027a6150>
> ```
> 
> Remember to use `asyncio.run()` to actually force execution by scheduling the `main()` coroutine (future object) for execution on the event loop:
> 
> ```python
> asyncio.run(routine)
> Hello ...
> World!
> ```
> 
> (Other coroutines can be executed with `await`. It is typical to wrap just `main()` in `asyncio.run()`, and chained coroutines with `await` will be called from there.)
> 
> **#2: By default, an async IO event loop runs in a single thread and on a single CPU core.**
> 
> Usually, running one single-threaded event loop in one CPU core is more than sufficient. It is also possible to run event loops across multiple cores. Check out this talk by John Reese for more, and be warned that your laptop may spontaneously combust.
> 
> **#3: Event loops are pluggable.**
> 
> That is, you could, if you really wanted, write your own event loop implementation and have it run tasks just the same. This is wonderfully demonstrated in the uvloop package, which is an implementation of the event loop in Cython.
> 
> That is what is meant by the term “pluggable event loop”: you can use any working implementation of an event loop, unrelated to the structure of the coroutines themselves. The asyncio package itself ships with two different event loop implementations, with the default being based on the selectors module. (The second implementation is built for Windows only.)
> 

**Example**

当涉及到多个连接的情况时，我们可以创建多个客户端连接到服务器，并观察异步处理多个连接的过程。

服务器端代码 `server.py`：

```python
import asyncio

async def handle_client(reader, writer):
    while True:
        data = await reader.read(100)
        if not data:
            break
        message = data.decode()
        addr = writer.get_extra_info('peername')
        print(f"Received {message} from {addr}")
        writer.write(data)
        await writer.drain()
    writer.close()

async def main():
    server = await asyncio.start_server(handle_client, '127.0.0.1', 8888)
    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')
    async with server:
        await server.serve_forever()

asyncio.run(main())
```

客户端代码 `client.py`：

```python
import asyncio

async def send_message(reader, writer):
    writer.write(b"Hello from client!\\n")
    await writer.drain()
    data = await reader.read(100)
    print(f"Received: {data.decode()}")

async def main():
    for i in range(5):  # 创建5个客户端连接
        reader, writer = await asyncio.open_connection('127.0.0.1', 8888)
        await send_message(reader, writer)
        writer.close()
        await writer.wait_closed()

asyncio.run(main())
```

运行以上代码，我们会得到类似如下的输出：

在服务器端 `server.py` 中的输出：

```
Serving on ('127.0.0.1', 8888)
Received Hello from client! from ('127.0.0.1', 54778)
Received Hello from client! from ('127.0.0.1', 54780)
Received Hello from client! from ('127.0.0.1', 54782)
Received Hello from client! from ('127.0.0.1', 54784)
Received Hello from client! from ('127.0.0.1', 54786)
```

在客户端 `client.py` 中的输出：

```
Received: Hello from client!
Received: Hello from client!
Received: Hello from client!
Received: Hello from client!
Received: Hello from client!
```

从服务器端的输出可以看出，服务器成功接收了来自多个客户端的连接，并处理了它们发送的消息。由于使用了异步 I/O 操作，服务器可以同时处理多个连接而不会阻塞。客户端的输出显示了它们成功发送消息给服务器，并接收到了服务器的响应。

### IO密集同步异步图解

<figure markdown="span">
![IO密集 同步](Python异步/Untitled.png)
<figcaption>IO密集 同步</figcaption>
</figure>

<figure markdown="span">
![IO密集 多进程](Python异步/Untitled%201.png )
<figcaption>IO密集 多进程</figcaption>
</figure>

<figure markdown="span">
![IO密集 多线程](Python异步/Picture1.png)
<figcaption>IO密集 多线程</figcaption>
</figure>

<figure markdown="span">
![IO密集 AsyncIO版本](Python异步/Untitled%202.png)
<figcaption>IO密集 AsyncIO版本</figcaption>
</figure>

## Summary

| 特点 | 线程并行 | 进程并行 | Asyncio |
| --- | --- | --- | --- |
| 执行模型 | 多线程模型，每个线程执行一个任务 | 多进程模型，每个进程执行一个任务 | 单线程事件循环模型，通过异步执行任务 |
| 并行性 | 受到全局解释锁 (GIL) 限制，无法真正并行执行Python字节码 | 每个进程有独立的解释器和内存空间，可以真正并行执行Python字节码 | 单线程执行，通过事件循环实现任务的异步执行 |
| 资源消耗 | 较少的内存消耗，线程之间共享内存空间 | 较大的内存消耗，进程之间独立的内存空间 | 较少的内存消耗，单线程执行，无需创建额外的线程或进程 |
| 阻塞操作 | 阻塞操作会影响其他线程的执行 | 阻塞操作不会影响其他进程的执行 | 非阻塞操作，通过事件循环等待 I/O 事件的完成 |
| 异常处理 | 一个线程的异常可能会影响其他线程的执行 | 一个进程的异常不会影响其他进程的执行 | 异常处理通常基于协程，可以很好地控制异常的传播 |
| 适用场景 | I/O密集型任务、网络编程等 | CPU密集型任务、I/O密集型任务、并发编程 | I/O密集型任务、高并发的网络编程等 |

## Reference

1. [https://realpython.com/async-io-python/#async-io-design-patterns](https://realpython.com/async-io-python/#async-io-design-patterns)
2. [https://realpython.com/python-concurrency/](https://realpython.com/python-concurrency/)
3. [https://stackoverflow.com/questions/49005651/how-does-asyncio-actually-work/51116910#51116910](https://stackoverflow.com/questions/49005651/how-does-asyncio-actually-work/51116910#51116910)
4. [https://realpython.com/python-gil/](https://realpython.com/python-gil/)