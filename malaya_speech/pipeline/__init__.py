"""
Copyright (c) 2017, Continuum Analytics, Inc. and contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

Neither the name of Continuum Analytics nor the names of any contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

https://github.com/python-streamz/streamz/blob/master/LICENSE.txt
"""

# this pipeline fundamentally follow streamz, but we added checkpoint as the fundamental
# so we have to improve the baseline code

import functools
import logging
from collections import deque
from tornado.locks import Condition
from .orderset import OrderedWeakrefSet

_global_sinks = set()
logger = logging.getLogger(__name__)
zipping = zip


def identity(x):
    return x


class Pipeline(object):
    _graphviz_shape = 'ellipse'
    _graphviz_style = 'rounded,filled'
    _graphviz_fillcolor = 'white'
    _graphviz_orientation = 0

    str_list = ['func', 'predicate', 'n', 'interval']

    __name__ = 'pipeline'

    def __init__(self, upstream = None, upstreams = None, name = None):
        self.downstreams = OrderedWeakrefSet()
        if upstreams is not None:
            self.upstreams = list(upstreams)
        else:
            self.upstreams = [upstream]

        for upstream in self.upstreams:
            if upstream:
                upstream.downstreams.add(self)

        self.name = name
        self.checkpoint = {}

    def __str__(self):
        s_list = []
        if self.name:
            s_list.append('{}; {}'.format(self.name, self.__class__.__name__))
        else:
            s_list.append(self.__class__.__name__)

        for m in self.str_list:
            s = ''
            at = getattr(self, m, None)
            if at:
                if not callable(at):
                    s = str(at)
                elif hasattr(at, '__name__'):
                    s = getattr(self, m).__name__
                elif hasattr(at.__class__, '__name__'):
                    s = getattr(self, m).__class__.__name__
                else:
                    s = None
            if s:
                s_list.append('{}={}'.format(m, s))
        if len(s_list) <= 2:
            s_list = [term.split('=')[-1] for term in s_list]

        text = '<'
        text += s_list[0]
        if len(s_list) > 1:
            text += ': '
            text += ', '.join(s_list[1:])
        text += '>'
        return text

    __repr__ = __str__

    @classmethod
    def register_api(cls, modifier = identity):
        def _(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                return func(*args, **kwargs)

            setattr(cls, func.__name__, modifier(wrapped))
            return func

        return _

    def _climb(self, source, name, x):

        for upstream in list(source.upstreams):
            if not upstream:
                source.checkpoint[name] = x
            else:
                self._climb(upstream, name, x)

    def _emit(self, x):

        name = type(self).__name__
        if hasattr(self, 'func'):
            if 'lambda' in self.func.__name__:
                if not self.name:
                    name = self.func.__name__
                else:
                    name = ''
            else:
                name = self.func.__name__
        if self.name:
            if len(name):
                name = f'{name}_{self.name}'
            else:
                name = self.name

        self._climb(self, name, x)

        result = []
        for downstream in list(self.downstreams):
            r = downstream.update(x, who = self)
            if type(r) is list:
                result.extend(r)
            else:
                result.append(r)

        return [element for element in result if element is not None]

    def update(self, x, who = None):
        self._emit(x)

    @property
    def upstream(self):
        if len(self.upstreams) != 1:
            raise ValueError('Pipeline has multiple upstreams')
        else:
            return self.upstreams[0]

    def visualize(self, filename = 'pipeline.png', **kwargs):
        """
        Render the computation of this object's task graph using graphviz.

        Requires ``graphviz`` to be installed.

        Parameters
        ----------
        filename : str, optional
            The name of the file to write to disk.
        kwargs:
            Graph attributes to pass to graphviz like ``rankdir="LR"``
        """
        from .graph import visualize

        return visualize(self, filename, **kwargs)

    def emit(self, x):
        result = self._emit(x)
        self.checkpoint.pop('Pipeline', None)
        return self.checkpoint

    def __call__(self, x):
        return self.emit(x)


@Pipeline.register_api()
class map(Pipeline):
    """ 
    apply a function / method to the pipeline

    Examples
    --------
    >>> source = Pipeline()
    >>> source.map(lambda x: x + 1).map(print)
    >>> source.emit(1)
    2
    """

    def __init__(self, upstream, func, *args, **kwargs):
        self.func = func
        name = kwargs.pop('name', None)
        self.kwargs = kwargs
        self.args = args

        Pipeline.__init__(self, upstream, name = name)
        _global_sinks.add(self)

    def update(self, x, who = None):
        try:
            result = self.func(x, *self.args, **self.kwargs)
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return self._emit(result)


@Pipeline.register_api()
class batching(Pipeline):
    """ 
    Batching stream into tuples

    Examples
    --------
    >>> source = Pipeline()
    >>> source.batching(2).map(print)
    >>> source.emit([1,2,3,4,5])
    ([1, 2], [3, 4], [5])
    """

    _graphviz_shape = 'diamond'

    def __init__(self, upstream, n, **kwargs):
        self.n = n
        Pipeline.__init__(self, upstream, **kwargs)

        _global_sinks.add(self)

    def update(self, x, who = None):
        result = []
        for i in range(0, len(x), self.n):
            index = min(i + self.n, len(x))
            result.append(x[i:index])
        return self._emit(tuple(result))


@Pipeline.register_api()
class partition(Pipeline):
    """ 
    Partition stream into tuples of equal size

    Examples
    --------
    >>> source = Pipeline()
    >>> source.partition(3).map(print)
    >>> for i in range(10):
    ...     source.emit(i)
    (0, 1, 2)
    (3, 4, 5)
    (6, 7, 8)
    """

    _graphviz_shape = 'diamond'

    def __init__(self, upstream, n, **kwargs):
        self.n = n
        self.buffer = []
        Pipeline.__init__(self, upstream, **kwargs)

        _global_sinks.add(self)

    def update(self, x, who = None):
        self.buffer.append(x)
        if len(self.buffer) == self.n:
            result, self.buffer = self.buffer, []
            return self._emit(tuple(result))
        else:
            return []


@Pipeline.register_api()
class sliding_window(Pipeline):
    """ 
    Produce overlapping tuples of size n

    Parameters
    ----------
    return_partial : bool
        If True, yield tuples as soon as any events come in, each tuple being
        smaller or equal to the window size. If False, only start yielding
        tuples once a full window has accrued.

    Examples
    --------
    >>> source = Pipeline()
    >>> source.sliding_window(3, return_partial=False).map(print)
    >>> for i in range(8):
    ...     source.emit(i)
    (0, 1, 2)
    (1, 2, 3)
    (2, 3, 4)
    (3, 4, 5)
    (4, 5, 6)
    (5, 6, 7)
    """

    _graphviz_shape = 'diamond'

    def __init__(self, upstream, n, return_partial = True, **kwargs):
        self.n = n
        self.buffer = deque(maxlen = n)
        self.partial = return_partial
        Pipeline.__init__(self, upstream, **kwargs)

        _global_sinks.add(self)

    def update(self, x, who = None):
        self.buffer.append(x)
        if self.partial or len(self.buffer) == self.n:
            return self._emit(tuple(self.buffer))
        else:
            return []


@Pipeline.register_api()
class foreach_map(Pipeline):
    """ 
    Apply a function to every element in a tuple in the stream.

    Parameters
    ----------
    func: callable
    method: str, optional (default='sync')
        method to process each elements.

        * ``'sync'`` - loop one-by-one to process.
        * ``'async'`` - async process all elements at the same time.
        * ``'thread'`` - multithreading level to process all elements at the same time. 
                         Default is 1 worker. Override `worker_size=n` to increase.
        * ``'process'`` - multiprocessing level to process all elements at the same time. 
                          Default is 1 worker. Override `worker_size=n` to increase.

    *args :
        The arguments to pass to the function.
    **kwargs:
        Keyword arguments to pass to func.

    Examples
    --------
    >>> source = Pipeline()
    >>> source.foreach_map(lambda x: 2*x).map(print)
    >>> for i in range(3):
    ...     source.emit((i, i))
    (0, 0)
    (2, 2)
    (4, 4)
    """

    def __init__(self, upstream, func, method = 'sync', *args, **kwargs):
        method = method.lower()
        if method not in ['sync', 'async', 'thread', 'process']:
            raise ValueError(
                'method only supported [`sync`, `async`, `thread`, `process`]'
            )
        self.func = func
        name = kwargs.pop('name', None)
        worker_size = kwargs.pop('worker_size', 1)
        self.worker_size = worker_size
        self.kwargs = kwargs
        self.args = args
        self.method = method
        if self.method == 'async':
            try:
                from tornado import gen
            except:
                raise ValueError(
                    'tornado not installed. Please install it by `pip install tornado` and try again.'
                )

        if self.method in ['thread', 'process']:
            try:
                import dask.bag as db
            except:
                raise ValueError(
                    'dask not installed. Please install it by `pip install dask` and try again.'
                )

        Pipeline.__init__(self, upstream, name = name)
        _global_sinks.add(self)

    def update(self, x, who = None):
        try:
            if self.method == 'async':
                from tornado import gen

                @gen.coroutine
                def function(e, *args, **kwargs):
                    return self.func(e, *args, **kwargs)

                @gen.coroutine
                def loop():
                    r = yield [
                        function(e, *self.args, **self.kwargs) for e in x
                    ]
                    return r

                result = loop().result()

            if self.method in ['thread', 'process']:
                import dask.bag as db

                bags = db.from_sequence(x)
                mapped = bags.map(self.func, *self.args, **self.kwargs)

                if self.method == 'thread':
                    scheduler = 'threads'

                if self.method == 'process':
                    scheduler = 'processes'

                result = mapped.compute(
                    scheduler = scheduler, num_workers = self.worker_size
                )
            if self.method == 'sync':
                result = [self.func(e, *self.args, **self.kwargs) for e in x]
        except Exception as e:
            logger.exception(e)
            raise
        else:
            return self._emit(result)


@Pipeline.register_api()
class flatten(Pipeline):
    """ 
    Flatten streams of lists or iterables into a stream of elements

    Examples
    --------
    >>> source = Pipeline()
    >>> source.flatten().map(print)
    >>> source.emit([[1, 2, 3], [4, 5], [6, 7, 7]])
    [1, 2, 3, 4, 5, 6, 7, 7]

    """

    def __init__(self, upstream, *args, **kwargs):
        name = kwargs.pop('name', None)
        self.kwargs = kwargs
        self.args = args

        Pipeline.__init__(self, upstream, name = name)
        _global_sinks.add(self)

    def update(self, x, who = None):
        L = []
        for item in x:
            if isinstance(item, list) or isinstance(item, tuple):
                L.extend(item)
            else:
                L.append(item)
        return self._emit(L)


@Pipeline.register_api()
class zip(Pipeline):
    """
    Combine 2 branches into 1 branch.

    Examples
    --------
    >>> source = Pipeline()
    >>> left = source.map(lambda x: x + 1, name = 'left')
    >>> right = source.map(lambda x: x + 10, name = 'right')
    >>> left.zip(right).map(sum).map(print)
    >>> source.emit(2)
    15
    """

    _graphviz_orientation = 270
    _graphviz_shape = 'triangle'

    def __init__(self, *upstreams, **kwargs):
        self.maxsize = kwargs.pop('maxsize', 10)
        self.condition = Condition()
        self.literals = [
            (i, val)
            for i, val in enumerate(upstreams)
            if not isinstance(val, Pipeline)
        ]

        self.buffers = {
            upstream: deque()
            for upstream in upstreams
            if isinstance(upstream, Pipeline)
        }

        upstreams2 = [
            upstream for upstream in upstreams if isinstance(upstream, Pipeline)
        ]

        Pipeline.__init__(self, upstreams = upstreams2, **kwargs)
        _global_sinks.add(self)

    def _add_upstream(self, upstream):
        # Override method to handle setup of buffer for new stream
        self.buffers[upstream] = deque()
        super(zip, self)._add_upstream(upstream)

    def _remove_upstream(self, upstream):
        # Override method to handle removal of buffer for stream
        self.buffers.pop(upstream)
        super(zip, self)._remove_upstream(upstream)

    def pack_literals(self, tup):
        """ Fill buffers for literals whenever we empty them """
        inp = list(tup)[::-1]
        out = []
        for i, val in self.literals:
            while len(out) < i:
                out.append(inp.pop())
            out.append(val)

        while inp:
            out.append(inp.pop())

        return out

    def update(self, x, who = None):
        L = self.buffers[who]  # get buffer for stream
        L.append(x)
        if len(L) == 1 and all(self.buffers.values()):
            tup = tuple(self.buffers[up][0] for up in self.upstreams)
            for buf in self.buffers.values():
                buf.popleft()
            self.condition.notify_all()
            if self.literals:
                tup = self.pack_literals(tup)

            return self._emit(tup)
        elif len(L) > self.maxsize:
            return self.condition.wait()


@Pipeline.register_api()
class foreach_zip(Pipeline):

    _graphviz_orientation = 270
    _graphviz_shape = 'triangle'

    def __init__(self, *upstreams, **kwargs):
        self.maxsize = kwargs.pop('maxsize', 10)
        self.condition = Condition()
        self.literals = [
            (i, val)
            for i, val in enumerate(upstreams)
            if not isinstance(val, Pipeline)
        ]

        self.buffers = {
            upstream: deque()
            for upstream in upstreams
            if isinstance(upstream, Pipeline)
        }

        upstreams2 = [
            upstream for upstream in upstreams if isinstance(upstream, Pipeline)
        ]

        Pipeline.__init__(self, upstreams = upstreams2, **kwargs)
        _global_sinks.add(self)

    def _add_upstream(self, upstream):
        # Override method to handle setup of buffer for new stream
        self.buffers[upstream] = deque()
        super(zip, self)._add_upstream(upstream)

    def _remove_upstream(self, upstream):
        # Override method to handle removal of buffer for stream
        self.buffers.pop(upstream)
        super(zip, self)._remove_upstream(upstream)

    def pack_literals(self, tup):
        """ Fill buffers for literals whenever we empty them """
        inp = list(tup)[::-1]
        out = []
        for i, val in self.literals:
            while len(out) < i:
                out.append(inp.pop())
            out.append(val)

        while inp:
            out.append(inp.pop())

        return out

    def update(self, x, who = None):
        L = self.buffers[who]  # get buffer for stream
        L.append(x)
        if len(L) == 1 and all(self.buffers.values()):
            tup = tuple(self.buffers[up][0] for up in self.upstreams)
            for buf in self.buffers.values():
                buf.popleft()
            self.condition.notify_all()
            if self.literals:
                tup = self.pack_literals(tup)

            tup = tuple(zipping(*tup))
            return self._emit(tup)
        elif len(L) > self.maxsize:
            return self.condition.wait()
