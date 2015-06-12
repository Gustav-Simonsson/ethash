/*
	This file is part of go-ethereum

	go-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU Lesser General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	go-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU Lesser General Public License
	along with go-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/**
 * @authors
 * 	Gustav Simonsson <gustav.simonsson@gmail.com>
 * @date 2015
 *
 */

/*

  In summary, this code have two main parts:

  1. The NewOpenCL(...)  function configures a OpenCL device (for now only GPU) and
     loads the Ethash DAG into device memory

  2. The Search(...) function loads a Ethash nonce into device memory and
     executes the Ethash OpenCL kernel.

  Throughout the code, we refer to "host memory" and "device memory".
  For most systems (e.g. regular PC GPU miner) the host memory is RAM and
  device memory is the GPU global memory (e.g. GDDR5).

  References are refered to by [1], [3] etc.

  1. https://github.com/ethereum/wiki/wiki/Ethash
  2. https://github.com/ethereum/cpp-ethereum/blob/develop/libethash-cl/ethash_cl_miner.cpp
  3. https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/
  4. http://amd-dev.wpengine.netdna-cdn.com/wordpress/media/2013/12/AMD_OpenCL_Programming_User_Guide.pdf

*/

package ethash

//#cgo LDFLAGS: -w
//#include <stdint.h>
//#include <string.h>
//#include "src/libethash/internal.h"
import "C"

import (
	crand "crypto/rand"
	"encoding/binary"
	"fmt"
	"math"
	"math/big"
	mrand "math/rand"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/Gustav-Simonsson/go-opencl/cl"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/pow"
)

type OpenCL struct {
	ethash       *Ethash // Ethash full DAG & cache in host mem
	dagSize      uint64
	dagChunksNum uint64
	dagChunks    []*cl.MemObject // DAG in device mem

	headerBuff    *cl.MemObject // Hash of block-to-mine in device mem
	searchBuffers []*cl.MemObject

	searchKernel *cl.Kernel
	hashKernel   *cl.Kernel

	queue         *cl.CommandQueue
	ctx           *cl.Context
	device        *cl.Device
	workGroupSize int

	nonceRand *mrand.Rand
	result    common.Hash

	hashRate int32 // Go atomics & uint64 have some issues, int32 is supported on all platforms
}

const (
	SIZEOF_UINT32 = 4

	// See [1]
	ethashMixBytesLen = 128
	ethashAccesses    = 64

	// See [4]
	workGroupSize    = 32 // must be multiple of 8
	maxSearchResults = 63
	searchBuffSize   = 2
	globalWorkSize   = 1024 * 256

	//gpuMemMargin = 1024 * 1024 * 512

	// TODO: config flags for these
	//checkGpuMemMargin = true
)

/* See [2]. We basically do the same here, but the Go OpenCL bindings
   are at a slightly higher abtraction level.
*/
// TODO: proper solution for automatic DAG switch at epoch change
func NewOpenCL(blockNum uint64, dagChunksNum uint64) (*OpenCL, error) {
	if !(dagChunksNum == 1 || dagChunksNum == 4) {
		return nil, fmt.Errorf("DAG chunks num must be 1 or 4")
	}

	platforms, err := cl.GetPlatforms()
	if err != nil {
		return nil, fmt.Errorf("OpenCL GetPlaforms error (check your OpenCL installation) :", err)
	}

	// TODO: platform and device selection logic,
	//       for now use first platform and first GPU device
	platform := platforms[0]
	devices, err := cl.GetDevices(platform, cl.DeviceTypeGPU)
	if err != nil {
		return nil, fmt.Errorf("OpenCL GetDevices error (check your OpenCL installation) :", err)
	}

	device := devices[0]
	devMaxAlloc := uint64(device.MaxMemAllocSize())
	devGlobalMem := uint64(device.GlobalMemSize())
	deviceVersion := device.Version()

	fmt.Println("=========== OpenCL initialisation ===========")
	fmt.Println("Selected platform       ", platform.Name())
	fmt.Println("Selected device         ", device.Name())
	fmt.Println("Vendor                  ", device.Vendor())
	fmt.Println("Version                 ", deviceVersion)
	fmt.Println("Driver version          ", device.DriverVersion())
	fmt.Println("Address bits            ", device.AddressBits())
	fmt.Println("Max clock freq          ", device.MaxClockFrequency())
	fmt.Println("Global mem size         ", devGlobalMem)
	fmt.Println("Max constant buffer size", device.MaxConstantBufferSize())
	fmt.Println("Max mem alloc size      ", devMaxAlloc)
	fmt.Println("Max compute units       ", device.MaxComputeUnits())
	fmt.Println("Max work group size     ", device.MaxWorkGroupSize())
	fmt.Println("Max work item sizes     ", device.MaxWorkItemSizes())

	// TODO: support OpenCL 1.1,  2.0 ?
	// TODO: more fine grained version logic
	if !strings.Contains(deviceVersion, "OpenCL 1.2") {
		fmt.Println("Device must be of OpenCL version 1.2")
		return nil, fmt.Errorf("opencl version not supported")
	}

	fmt.Println("===============================================")

	pow := New()
	dagSize := pow.getDAGSize(blockNum)

	if dagSize > devGlobalMem {
		fmt.Printf("Warning: device memory may be insufficient: %v. DAG size: %v.\n", devGlobalMem, dagSize)
		fmt.Printf("You may have to run with -- --opencl-mem-chunking \n")
		// TODO: we continue since it seems sometimes clGetDeviceInfo reports wrong numbers
		//return nil, fmt.Errorf("Insufficient device memory")
	}

	if dagSize > devMaxAlloc {
		fmt.Printf("Warning: DAG size (%v) larger than device max memory allocation size (%v).\n", dagSize, devMaxAlloc)
		fmt.Printf("You may have to run with --opencl-mem-chunking\n")
		//return nil, fmt.Errorf("Insufficient device memory")
	}

	// generates DAG if we don't have it
	_ = pow.getDAG(blockNum)

	// and cache. TODO: unfuck
	pow.Light.getCache(blockNum)

	context, err := cl.CreateContext([]*cl.Device{device})
	if err != nil {
		return nil, fmt.Errorf("failed creating context:", err)
	}

	// TODO: test running with CL_QUEUE_PROFILING_ENABLE for profiling?
	queue, err := context.CreateCommandQueue(device, 0)
	if err != nil {
		return nil, fmt.Errorf("command queue err:", err)
	}

	// See [4] section 3.2 and [3] "clBuildProgram".
	// The OpenCL kernel code is compiled at run-time.
	kvs := make(map[string]string, 4)
	kvs["GROUP_SIZE"] = strconv.FormatUint(workGroupSize, 10)
	kvs["DAG_SIZE"] = strconv.FormatUint(dagSize/ethashMixBytesLen, 10)
	kvs["ACCESSES"] = strconv.FormatUint(ethashAccesses, 10)
	kvs["MAX_OUTPUTS"] = strconv.FormatUint(maxSearchResults, 10)
	kernelCode := replaceWords(kernel, kvs)

	program, err := context.CreateProgramWithSource([]string{kernelCode})
	if err != nil {
		return nil, fmt.Errorf("program err:", err)
	}

	/* if using AMD OpenCL impl, you can set this to debug on x86 CPU device.
	   see AMD OpenCL programming guide section 4.2

	   export in shell before running:
	   export AMD_OCL_BUILD_OPTIONS_APPEND="-g -O0"
	   export CPU_MAX_COMPUTE_UNITS=1

	buildOpts := "-g -cl-opt-disable"

	*/
	buildOpts := ""
	err = program.BuildProgram([]*cl.Device{device}, buildOpts)
	if err != nil {
		return nil, fmt.Errorf("program build err:", err)
	}

	var searchKernelName, hashKernelName string
	if dagChunksNum == 4 {
		searchKernelName = "ethash_search_chunks"
		hashKernelName = "ethash_hash_chunks"
	} else {
		searchKernelName = "ethash_search"
		hashKernelName = "ethash_hash"
	}

	searchKernel, err := program.CreateKernel(searchKernelName)
	hashKernel, err := program.CreateKernel(hashKernelName)
	if err != nil {
		return nil, fmt.Errorf("kernel err:", err)
	}

	// TODO: in case chunk allocation is not default when this DAG size appears, patch
	// the Go bindings (context.go) to work with uint64 as size_t
	if dagSize > math.MaxInt32 {
		fmt.Println("DAG too large for non-chunk allocation. Try running with --opencl-mem-chunking")
		return nil, fmt.Errorf("DAG too large for non-chunk alloc")
	}

	chunkSize := func(i uint64) uint64 {
		if dagChunksNum == 1 {
			return dagSize
		} else {
			if i == (dagChunksNum - 1) {
				return dagSize - ((dagChunksNum - 1) * (dagSize / dagChunksNum))
			} else {
				return dagSize / dagChunksNum
			}
		}
	}

	// allocate device mem
	dagChunks := make([]*cl.MemObject, dagChunksNum)
	for i := uint64(0); i < dagChunksNum; i++ {
		// TODO: patch up Go bindings to work with size_t, chunkSize will overflow if > maxint32
		dagChunk, err := context.CreateEmptyBuffer(cl.MemReadOnly, int(chunkSize(i)))
		if err != nil {
			return nil, fmt.Errorf("allocating dag chunks failed: ", err)
		}
		dagChunks[i] = dagChunk
	}

	// write DAG to device mem
	var offset uint64
	for i := uint64(0); i < dagChunksNum; i++ {
		offset = chunkSize(i) * i
		// TODO: fuck. this shit's gonna overflow some day
		dagOffsetPtr := unsafe.Pointer(uintptr(unsafe.Pointer(pow.Full.current.ptr.data)) + uintptr(offset))
		fmt.Println("offset, chunkSize, dagSize: ", offset, chunkSize(i), dagSize)
		//spew.Dump(C.GoBytes(dag.Ptr(), 128))
		//spew.Dump(C.GoBytes(unsafe.Pointer(dagChunk), 128))
		// offset into device buffer is always 0, offset into DAG depends on chunk
		_, err = queue.EnqueueWriteBuffer(dagChunks[i], true, 0, int(chunkSize(i)), dagOffsetPtr, nil)
		if err != nil {
			return nil, fmt.Errorf("writing to dag chunk failed: ", err)
		}
		//spew.Dump(C.GoBytes(unsafe.Pointer(dagChunk), 128))
	}

	searchBuffers := make([]*cl.MemObject, searchBuffSize)
	for i := 0; i < searchBuffSize; i++ {
		searchBuff, err := context.CreateEmptyBuffer(cl.MemWriteOnly, (1+maxSearchResults)*SIZEOF_UINT32)
		if err != nil {
			return nil, fmt.Errorf("search buffer err:", err)
		}
		searchBuffers[i] = searchBuff
	}

	// hash of block-to-mine in device mem
	headerBuff, err := context.CreateEmptyBuffer(cl.MemReadOnly, 32)
	if err != nil {
		return nil, fmt.Errorf("header buffer err:", err)
	}

	// Unique, random nonces are crucial for mining efficieny.
	// While we do not need cryptographically secure PRNG for nonces,
	// we want to have uniform distribution and minimal repetition of nonces.
	// We could guarantee strict uniqueness of nonces by generating unique ranges,
	// but a int64 seed from crypto/rand should be good enough.
	// we then use math/rand for speed and to avoid draining OS entropy pool
	seed, err := crand.Int(crand.Reader, big.NewInt(math.MaxInt64))
	if err != nil {
		return nil, err
	}
	nonceRand := mrand.New(mrand.NewSource(seed.Int64()))

	return &OpenCL{
			ethash:        pow,
			dagSize:       dagSize,
			dagChunksNum:  dagChunksNum,
			dagChunks:     dagChunks,
			headerBuff:    headerBuff,
			searchBuffers: searchBuffers,

			searchKernel: searchKernel,
			hashKernel:   hashKernel,

			queue:         queue,
			ctx:           context,
			device:        device,
			workGroupSize: workGroupSize,

			nonceRand: nonceRand,
		},
		nil
}

func (m *OpenCL) Search(block pow.Block, stop <-chan struct{}) (uint64, []byte) {
	headerHash := block.HashNoNonce()
	diff := block.Difficulty()
	target256 := new(big.Int).Div(MaxUint256, diff)
	target64 := new(big.Int).Rsh(target256, 192).Uint64()
	fmt.Println("target256, target64, diff ", target256, target64, diff)

	headerHashUSPtr := unsafe.Pointer(&headerHash[0])
	_, err := m.queue.EnqueueWriteBuffer(m.headerBuff, false, 0, 32, headerHashUSPtr, nil)
	if err != nil {
		return 0, []byte{0}
	}

	var zero uint32 = 0
	for i := 0; i < searchBuffSize; i++ {
		_, err := m.queue.EnqueueWriteBuffer(m.searchBuffers[i], false, 0, 4, unsafe.Pointer(&zero), nil)
		if err != nil {
			return 0, []byte{0}
		}
	}

	// we wait for this one before returning
	var preReturnEvent *cl.Event
	preReturnEvent, err = m.ctx.CreateUserEvent()
	if err != nil {
		return 0, []byte{0}
	}
	_, err = m.queue.EnqueueBarrierWithWaitList([]*cl.Event{preReturnEvent})
	if err != nil {
		return 0, []byte{0}
	}

	// wait for all search buffers to complete
	m.queue.Finish()

	err = m.searchKernel.SetArg(1, m.headerBuff)
	if err != nil {
		return 0, []byte{0}
	}

	argPos := 2
	for i := uint64(0); i < m.dagChunksNum; i++ {
		err = m.searchKernel.SetArg(argPos, m.dagChunks[i])
		if err != nil {
			return 0, []byte{0}
		}
		argPos++
	}
	err = m.searchKernel.SetArg(argPos+1, target64)
	if err != nil {
		return 0, []byte{0}
	}
	err = m.searchKernel.SetArg(argPos+2, uint32(math.MaxUint32))
	if err != nil {
		return 0, []byte{0}
	}

	if err != nil {
		fmt.Println("Error in Search setup: ", err)
		return 0, []byte{0}
	}

	n, md := searchLoop(m, &target64, target256, &headerHash, preReturnEvent)
	return n, md
}

func searchLoop(m *OpenCL, target64 *uint64, target256 *big.Int, headerHash *common.Hash, preReturnEvent *cl.Event) (uint64, []byte) {

	buf := 0
	// we grab a single random nonce and sets this as argument to the kernel search function
	// the device will then add each local threads gid to the nonce, creating a unique nonce
	// for each device computing unit executing in parallel
	var err error
	headerHashCPtr := hashToH256(*headerHash)
	var checkNonce uint64
	initNonce := uint64(m.nonceRand.Int63())
	loops := int64(0)
	prevHashRate := int32(0)
	start := time.Now().UnixNano()
	for nonce := initNonce; ; nonce += uint64(globalWorkSize) {

		if (loops % (1 << 8)) == 0 {
			elapsed := time.Now().UnixNano() - start
			loops := (float64(1e9) / float64(elapsed)) * float64(loops)
			hashrateDiff := int32(loops) - prevHashRate
			prevHashRate = int32(loops)
			atomic.AddInt32(&m.hashRate, hashrateDiff)
			fmt.Println("loopRate: ", int64(atomic.LoadInt32(&m.hashRate)))
		}
		loops++

		err = m.searchKernel.SetArg(0, m.searchBuffers[buf])
		if err != nil {
			fmt.Println("Error in Search clSetKernelArg : ", err)
			return 0, []byte{0}
		}
		// TODO: only works with 1 or 4 currently
		var argPos int
		if m.dagChunksNum == 1 {
			argPos = 3
		} else if m.dagChunksNum == 4 {
			argPos = 6
		}
		err = m.searchKernel.SetArg(argPos, nonce)
		if err != nil {
			fmt.Println("Error in Search clSetKernelArg : ", err)
			return 0, []byte{0}
		}

		// executes kernel; either the "ethash_search" or "ethash_search_chunks" function
		_, err := m.queue.EnqueueNDRangeKernel(
			m.searchKernel,
			[]int{0},
			[]int{globalWorkSize},
			[]int{m.workGroupSize},
			nil)
		if err != nil {
			fmt.Println("Error in Search clEnqueueNDRangeKernel : ", err)
			return 0, []byte{0}
		}

		// TODO: why do we only check the last search buffer?
		buf = (buf + 1) % searchBuffSize

		if buf == 0 {
			//fmt.Println("FUNKY: ", buf, m.searchBuffers[buf])
			cres, _, err := m.queue.EnqueueMapBuffer(m.searchBuffers[0], true,
				cl.MapFlagRead, 0, (1+maxSearchResults)*SIZEOF_UINT32,
				nil)
			if err != nil {
				fmt.Println("Error in Search clEnqueueMapBuffer: ", err)
				return 0, []byte{0}
			}

			results := cres.ByteSlice()
			//fmt.Println("FUNKY: len ByteSlice", len(results))
			//fmt.Println("FUNKY: results", hex.EncodeToString(results))
			nfound := binary.BigEndian.Uint32(results)
			nfound = uint32(math.Min(float64(nfound), float64(maxSearchResults)))
			//nonces := make([]uint64, maxSearchResults)
			// OpenCL returns the offsets from the start nonce
			for i := uint32(0); i < nfound; i++ {
				lo := (i + 1) * SIZEOF_UINT32
				hi := (i + 2) * SIZEOF_UINT32
				upperNonce := uint64(binary.BigEndian.Uint32(results[lo:hi]))
				checkNonce = nonce + upperNonce
				//fmt.Println("FUNKY: upperNonce", upperNonce)
				//fmt.Println("FUNKY: checkNonce", checkNonce)
				//fmt.Println("FUNKY: i, n: ", i, checkNonce)
				if checkNonce != 0 {
					//fmt.Println("FUNKY2: i, n: ", i, checkNonce)
					cn := C.uint64_t(checkNonce)
					ds := C.uint64_t(m.dagSize)
					//fmt.Println("FUNKY: light compute args: ", m.ethash.Light.current.ptr, ds, hashToH256(headerHash), cn)
					// We verify that the nonce is indeed a solution by calling Ethash verification function
					// in C on the CPU.
					// TODO: if we're confident OpenCL always returns a valid nonce, we can skip this check
					// for performance
					ret := C.ethash_light_compute_internal(m.ethash.Light.current.ptr, ds, headerHashCPtr, cn)
					if ret.success && h256ToHash(ret.result).Big().Cmp(target256) <= 0 {
						// prioritise returning mined solution over errors
						_, err = m.queue.EnqueueUnmapMemObject(m.searchBuffers[buf], cres, nil)
						defer logErr(err)
						// TODO: return result before waiting here
						err = cl.WaitForEvents([]*cl.Event{preReturnEvent})
						defer logErr(err)

						fmt.Println("OpenCL Search returning: n, mix", checkNonce, C.GoBytes(unsafe.Pointer(&ret.mix_hash), C.int(32)))
						return checkNonce, C.GoBytes(unsafe.Pointer(&ret.mix_hash), C.int(32))
					}
				}
			}
			_, err = m.queue.EnqueueUnmapMemObject(m.searchBuffers[buf], cres, nil)
			if err != nil {
				fmt.Println("Error in Search clEnqueueUnMapMemObject: ", err)
				return 0, []byte{0}
			}
		}
	}
	err = cl.WaitForEvents([]*cl.Event{preReturnEvent})
	if err != nil {
		fmt.Println("Error in Search clWaitForEvents: ", err)
		return 0, []byte{0}
	}
	return 0, []byte{0}
}

func (m *OpenCL) Verify(block pow.Block) bool {
	return m.ethash.Light.Verify(block)
}
func (m *OpenCL) GetHashrate() int64 {
	// TODO: set in search loop
	return int64(atomic.LoadInt32(&m.hashRate))
}
func (m *OpenCL) Turbo(on bool) {
	// This is GPU mining. Always be turbo.
}

func replaceWords(text string, kvs map[string]string) string {
	for k, v := range kvs {
		text = strings.Replace(text, k, v, -1)
	}
	return text
}

func logErr(err error) {
	if err != nil {
		fmt.Println("Error in OpenCL call:", err)
	}
}

func argErr(err error) error {
	return fmt.Errorf("arg err: %v", err)
}
