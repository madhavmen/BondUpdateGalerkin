# `@lazy` is deliberately a syntactic, lexical opt-in.  Its only rewrites are
# contractions; structural operations propagate laziness from their inputs.
"""
    _lazy_finalize(x)

Finalize the value produced by an `@lazy` expression.

`x` may be an `AbstractTLArray`, in which case it is materialized as a concrete
`TLArray`, or a `Number`, which is returned unchanged. Any other value is
rejected because `@lazy` is intended for tensor expressions and scalar
contraction results, not as a general-purpose expression transformer.
"""
@inline _lazy_finalize(q::AbstractTLArray) = TLArray(q)
@inline _lazy_finalize(x::Number) = x
_lazy_finalize(x) = throw(ArgumentError(
    "@lazy must finish with an AbstractTLArray or Number, got $(typeof(x))"))

"""
    _lazy_callee(name_or_expr)

Choose the callee used when rewriting a call inside `@lazy`.

For a symbol callee, `contract` and binary/n-ary `*` are redirected to
`Telum._lazy_contract`; other callees are left unchanged. For qualified calls,
only `Telum.contract(...)` is redirected. This keeps the macro lexical and
conservative: unrelated user functions, methods from other modules, and
structural TLArray operations are not rewritten here.
"""
@inline _lazy_callee(name::Symbol) =
    name === :contract || name === :* ? GlobalRef(@__MODULE__, :_lazy_contract) : name

@inline function _lazy_callee(callee::Expr)
    callee.head === :. || return callee
    length(callee.args) == 2 || return callee
    callee.args[1] === :Telum || return callee
    name = callee.args[2]
    name isa QuoteNode || return callee
    name.value === :contract || return callee
    return GlobalRef(@__MODULE__, :_lazy_contract)
end

"""
    _lazy_rewrite(ex) -> Expr
    _lazy_rewrite(ex, in_nested_function::Bool) -> Any

Rewrite a parsed expression so eligible contractions become lazy contractions.

`ex` is an expression tree or literal node. `in_nested_function` tracks whether
the traversal is inside a nested function or anonymous function; top-level
`return` values are finalized, but nested-function returns are not captured by
the surrounding macro. Assignments rewrite only the right-hand side. Quoted,
inert, and macrocall expressions are intentionally left unchanged to avoid
rewriting code that Julia or another macro should own.

For call expressions, `_lazy_callee` selects the rewritten callee and arguments
are rewritten recursively. N-ary multiplication is lowered to a left-associated
chain of binary lazy contractions so the contraction builder only needs to
support two-input composition.
"""
_lazy_rewrite(ex) = _lazy_rewrite(ex, false)
_lazy_rewrite(ex, ::Bool) = ex

function _lazy_rewrite(ex::Expr, in_nested_function::Bool)
    ex.head === :quote && return ex
    ex.head === :inert && return ex
    ex.head === :macrocall && return ex
    if ex.head === :(=)
        return Expr(:(=), ex.args[1], _lazy_rewrite(ex.args[2], in_nested_function))
    elseif ex.head === :function || ex.head === :->
        args = copy(ex.args)
        args[end] = _lazy_rewrite(args[end], true)
        return Expr(ex.head, args...)
    elseif ex.head === :return && !in_nested_function
        value = length(ex.args) == 1 ? _lazy_rewrite(ex.args[1], false) : nothing
        return Expr(:return, Expr(:call, GlobalRef(@__MODULE__, :_lazy_finalize), value))
    elseif ex.head === :call
        callee = _lazy_callee(ex.args[1])
        args = map(arg -> _lazy_rewrite(arg, in_nested_function), ex.args[2:end])
        # Julia lowers `a * b * c` to one n-ary call; lazy contractions remain binary.
        if ex.args[1] === :* && length(args) > 2
            return foldl(args[2:end]; init=args[1]) do lhs, rhs
                Expr(:call, callee, lhs, rhs)
            end
        end
        return Expr(:call, callee, args...)
    end
    return Expr(ex.head, map(arg -> _lazy_rewrite(arg, in_nested_function), ex.args)...)
end

"""
    @lazy expression

Rewrite supported tensor operations in `expression` so eligible intermediate
results remain lazy until `materialize` or payload access requires evaluation.
The macro rewrites only recognized calls; unsupported
calls and nested function definitions are retained unchanged. Lazy wrappers
preserve source sector-slot numbering and carry deferred scale, conjugation,
and permutation state rather than transforming RMT storage eagerly.
"""
macro lazy(ex)
    rewritten = _lazy_rewrite(ex)
    return Expr(:call, GlobalRef(@__MODULE__, :_lazy_finalize), esc(rewritten))
end
