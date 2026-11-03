#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <cuda_runtime.h>

// Device memory a structure needs: persistent buffers and temporary scratch.
struct SpatialSizes {
    size_t nodes    = 0;
    size_t prim_ids = 0;
    size_t scratch  = 0;

    size_t total() const { return nodes + prim_ids + scratch; }

    size_t persistent() const { return nodes + prim_ids; }
};

// Carves aligned sub-buffers out of one scratch allocation. With a null base it
// only measures, so the same code computes the size and the layout.
class ScratchLayout {
public:

    static constexpr size_t ALIGN = 256;

    ScratchLayout() = default;
    explicit ScratchLayout(void* base) : base_(static_cast<unsigned char*>(base)) {}

    template<typename T>
    T* take(size_t count)
    {
        used_ = align_up(used_);
        const size_t off = used_;
        used_ += count * sizeof(T);
        return base_ ? reinterpret_cast<T*>(base_ + off) : nullptr;
    }

    void* take_bytes(size_t bytes)
    {
        used_ = align_up(used_);
        const size_t off = used_;
        used_ += bytes;
        return base_ ? static_cast<void*>(base_ + off) : nullptr;
    }

    size_t total() const { return align_up(used_); }

    bool measuring() const { return base_ == nullptr; }

private:
    static size_t align_up(size_t v) { return (v + ALIGN - 1) / ALIGN * ALIGN; }

    unsigned char* base_ = nullptr;
    size_t         used_ = 0;
};

// A structure satisfies the contract if it has static reserve(n) returning SpatialSizes.
namespace contract {

template<typename, typename = void>
struct satisfies : std::false_type {};

template<typename DS>
struct satisfies<DS, std::void_t<decltype(DS::reserve(uint32_t{}))>>
    : std::is_same<decltype(DS::reserve(uint32_t{})), SpatialSizes> {};

}

template<typename DS>
using satisfies_contract = contract::satisfies<DS>;
