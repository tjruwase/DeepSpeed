/*
Copyright 2020 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include <memory>
#include <queue>
#include "deepspeed_py_aio.h"

struct io_op_desc_t {
    const bool _read_op;
    torch::Tensor _buffer;
    int _fd;
    const std::string _filename;
    const long long int _file_num_bytes;
    const int _num_threads;
    const int _num_bytes_per_thread;
    torch::Tensor _contiguous_buffer;
    const bool _validate;

    io_op_desc_t(const bool read_op,
                 const torch::Tensor& buffer,
                 const int fd,
                 const char* filename,
                 const long long int file_num_bytes,
                 const int num_threads,
                 const bool validate);

    virtual void run(const int tid,
             std::unique_ptr<aio_context>& aio_ctxt,
             deepspeed_aio_config_t* aio_config);

    virtual char* data_ptr() const;

    virtual void validate();

    virtual void fini();
};