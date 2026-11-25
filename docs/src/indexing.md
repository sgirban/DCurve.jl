# Smart Point Indexing

`DiscreteCurve` supports zero-copy point extraction, `end` arithmetic, index collections, and automatic periodic wrap-around on closed curves (`c[0] == c[end]` and `c[end + 1] == c[1]`).

```@example indexing
using DCurve

# Open curve: standard 1-based bounds checking
c_open = DiscreteCurve([0.0 1.0 2.0; 0.0 2.0 0.0]; closed = false)
c_open[end - 1]

# Closed curve: modular wrap-around via mod1
c_closed = DiscreteCurve([0.0 1.0 2.0; 0.0 2.0 0.0]; closed = true)
(c_closed[0] == c_closed[end], c_closed[end + 1] == c_closed[1])
```

## API Reference

```@docs
Base.getindex(::DiscreteCurve, ::Integer)
```