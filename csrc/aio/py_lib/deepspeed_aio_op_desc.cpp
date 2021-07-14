/*
Copyright 2020 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_aio_op_desc.h"

using namespace std;

io_op_desc_t::io_op_desc_t(const bool read_op,
                           const torch::Tensor& buffer,
                           const int fd,
                           const char* filename,
                           const long long int file_num_bytes,
                           const int num_threads,
                           const bool validate)
    : _read_op(read_op),
      _buffer(buffer),
      _fd(fd),
      _filename(filename),
      _file_num_bytes(file_num_bytes),
      _num_threads(num_threads),
      _num_bytes_per_thread(file_num_bytes/num_threads),
      _validate(validate)
{
    _contiguous_buffer = _buffer.contiguous();
}

char* io_op_desc_t::data_ptr() const { return (char*)_contiguous_buffer.data_ptr(); }

void io_op_desc_t::fini()
{
}


void io_op_desc_t::validate()
{
    validate_aio_operation(_read_op,
                           _filename.c_str(),
                           data_ptr(),
                           _file_num_bytes);
}

void io_op_desc_t::run(const int tid,
                       std::unique_ptr<aio_context>& aio_ctxt,
                       deepspeed_aio_config_t* aio_config)
{
    assert (tid < _num_threads);
    const auto base_offset = _num_bytes_per_thread * tid;

    std::unique_ptr<io_xfer_ctxt> xfer_ctxt(new io_xfer_ctxt(
        _fd, base_offset, _num_bytes_per_thread, data_ptr()));

    if (aio_config->_overlap_events) {
        do_aio_operation_overlap(
            _read_op, aio_ctxt, xfer_ctxt, aio_config, nullptr);
    } else {
        do_aio_operation_sequential(
            _read_op, aio_ctxt, xfer_ctxt, aio_config, nullptr);
    }
}
