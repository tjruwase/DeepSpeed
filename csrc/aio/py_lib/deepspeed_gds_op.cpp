/*
Copyright 2020 The Microsoft DeepSpeed Team
Licensed under the MIT license.

Functionality for swapping optimizer tensors to/from (NVMe) storage devices.
*/

#include "deepspeed_gds_op.h"
#include <cstdlib>
#include <set>

using namespace std;

static std::set<char*> s_buffer_registry;

void _safe_handle_register(const int fd, CUfileDescr_t& cf_descr, CUfileHandle_t& cf_handle)
{
    memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileError_t status = cuFileHandleRegister(&cf_handle, &cf_descr);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "file register error:" << cuFileGetErrorString(status) << std::endl;
        close(fd);
        exit(EXIT_FAILURE);
    }
}

void _safe_buffer_register(void* gpu_buffer, const size_t size, const int64_t device)
{
    // std::cout << "gpu: " << device  << " gds register buffer " << gpu_buffer << " of size " <<
    // size << std::endl;
    CUfileError_t status = cuFileBufRegister(gpu_buffer, size, 0);
    if (status.err != CU_FILE_SUCCESS) {
        std::cout << "buffer register failed:" << cuFileGetErrorString(status) << std::endl;
        //exit(EXIT_FAILURE);
    }
}

gds_op_desc_t::gds_op_desc_t(const bool read_op,
                             const torch::Tensor& buffer,
                             const int fd,
                             const char* filename,
                             const long long int file_num_bytes,
                             const int num_threads,
                             const bool validate)
    : io_op_desc_t(read_op, buffer, fd, filename, file_num_bytes, num_threads, validate)
{
    assert(_buffer.is_cuda());
    _contiguous_buffer = _buffer.contiguous();

    check_cudaruntimecall(cudaSetDevice(_buffer.get_device()));

    _safe_handle_register(fd, _cf_descr, _cf_handle);

//    const auto search = s_buffer_registry.find(data_ptr());
//    if (search == s_buffer_registry.end()) {
//        _safe_buffer_register(
//            _contiguous_buffer.data_ptr(), _buffer.nbytes(), _buffer.get_device());
//        s_buffer_registry.insert(data_ptr());
//    } else {
//        // std::cout << "gpu: " << _buffer.get_device() << " gds skip register buffer " <<
//        // _contiguous_buffer.data_ptr() << " of size " << _buffer.nbytes() << std::endl;
//    }
}

char* gds_op_desc_t::data_ptr() const { return (char*)_contiguous_buffer.data_ptr(); }

void gds_op_desc_t::fini()
{
    // check_cuFileCall(cuFileBufDeregister(_buffer.data_ptr()), "file buffer deregister");

    cuFileHandleDeregister(_cf_handle);
}

void gds_op_desc_t::validate()
{
    check_cudaruntimecall(cudaSetDevice(_buffer.get_device()));
    const auto cpu_buffer = _buffer.to(torch::kCPU);
    validate_aio_operation(
        _read_op, _filename.c_str(), (char*)(cpu_buffer.data_ptr()), _file_num_bytes);
}

void gds_op_desc_t::run(const int tid,
                        std::unique_ptr<aio_context>& aio_ctxt,
                        deepspeed_aio_config_t* aio_config)
{
    assert(tid < _num_threads);
    check_cudaruntimecall(cudaSetDevice(_buffer.get_device()));
    if (_read_op) {
        _read_file(tid);
    } else {
        _write_file(tid);
    }
}

void gds_op_desc_t::_report_error(const ssize_t return_code,
                                  const int error_num,
                                  const off_t offset)
{
    const auto op_string = _read_op ? "read failed with " : "write failed with ";
    const auto error_string = IS_CUFILE_ERR(return_code) ? "cuFile error: " : "posix error: ";
    const auto error_code = IS_CUFILE_ERR(return_code) ? cuFileGetErrorString(return_code)
                                                       : cuFileGetErrorString(error_num);
    std::cerr << op_string << error_string << error_code << " return code = " << return_code
              << " filename = " << _filename.c_str() << " num bytes = " << _num_bytes_per_thread
              << " offset = " << offset << std::endl;
    exit(EXIT_FAILURE);
}

void gds_op_desc_t::_read_file(const int tid)
{
    const auto base_offset = _num_bytes_per_thread * tid;
    auto ret = cuFileRead(_cf_handle, data_ptr(), _num_bytes_per_thread, base_offset, base_offset);
    if (ret < 0) { _report_error(ret, errno, base_offset); }
}

void gds_op_desc_t::_write_file(const int tid)
{
    const auto base_offset = _num_bytes_per_thread * tid;
    auto ret = cuFileWrite(_cf_handle, data_ptr(), _num_bytes_per_thread, base_offset, base_offset);
    if (ret < 0) { _report_error(ret, errno, base_offset); }
}
