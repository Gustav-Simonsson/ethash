package ethash

import (
	"bytes"
	"encoding/hex"
	"log"
	"math/big"
	"strconv"
	"sync"
	"testing"

	"github.com/ethereum/go-ethereum/common"
)

type testBlock struct {
	difficulty  *big.Int
	hashNoNonce common.Hash
	nonce       uint64
	mixDigest   common.Hash
	number      uint64
}

func (b *testBlock) Difficulty() *big.Int     { return b.difficulty }
func (b *testBlock) HashNoNonce() common.Hash { return b.hashNoNonce }
func (b *testBlock) Nonce() uint64            { return b.nonce }
func (b *testBlock) MixDigest() common.Hash   { return b.mixDigest }
func (b *testBlock) NumberU64() uint64        { return b.number }

func TestEthash(t *testing.T) {
	block := &testBlock{difficulty: big.NewInt(10)}
	eth := NewForTesting()
	nonce, _ := eth.Search(block, nil)
	block.nonce = nonce

	// Verify the block concurrently to check for data races.
	var wg sync.WaitGroup
	wg.Add(100)
	for i := 0; i < 100; i++ {
		go func() {
			if !eth.Verify(block) {
				t.Error("Block could not be verified")
			}
			wg.Done()
		}()
	}
	wg.Wait()
}

func TestGetSeedHash(t *testing.T) {
	seed0, err := GetSeedHash(0)
	if err != nil {
		t.Errorf("Failed to get seedHash for block 0: %v", err)
	}
	if bytes.Compare(seed0, make([]byte, 32)) != 0 {
		log.Printf("seedHash for block 0 should be 0s, was: %v\n", seed0)
	}
	seed1, err := GetSeedHash(30000)
	if err != nil {
		t.Error(err)
	}

	// From python:
	// > from pyethash import get_seedhash
	// > get_seedhash(30000)
	expectedSeed1, err := hex.DecodeString("290decd9548b62a8d60345a988386fc84ba6bc95484008f6362f93160ef3e563")
	if err != nil {
		t.Error(err)
	}

	if bytes.Compare(seed1, expectedSeed1) != 0 {
		log.Printf("seedHash for block 1 should be: %v,\nactual value: %v\n", expectedSeed1, seed1)
	}
}

func TestEthashRealParams(t *testing.T) {
	h, _ := hex.DecodeString("ab4d3689b354bbc3ce5ae4e38e821573cc35ecbb50962bc7b9b9eff7f405d5ed")
	md, _ := hex.DecodeString("122801f412807c6578082a1bc3d17c82e5454a2951338d444e5adb73582987c2")
	//d := new(big.Int).Exp(big.NewInt(2), big.NewInt(254), big.NewInt(0))
	d, _ := big.NewInt(0).SetString("0x020000", 0)
	n, _ := strconv.ParseUint("1dc31552da67be0a", 16, 64)
	block := &testBlock{
		difficulty:  d,
		hashNoNonce: common.BytesToHash(h),
		mixDigest:   common.BytesToHash(md),
		nonce:       n,
		number:      1,
	}

	eth := New()
	//nonce, _ := eth.Search(block, nil)
	//block.nonce = nonce

	if !eth.Verify(block) {
		t.Error("Block could not be verified")
	}
}

func TestEthashRealParamsNoNonce(t *testing.T) {
	h, _ := hex.DecodeString("ab4d3689b354bbc3ce5ae4e38e821573cc35ecbb50962bc7b9b9eff7f405d5ed")
	md, _ := hex.DecodeString("122801f412807c6578082a1bc3d17c82e5454a2951338d444e5adb73582987c2")
	//d := new(big.Int).Exp(big.NewInt(2), big.NewInt(254), big.NewInt(0))
	d, _ := big.NewInt(0).SetString("0x020000", 0)
	//n, _ := strconv.ParseUint("1dc31552da67be0a", 16, 64)
	block := &testBlock{
		difficulty:  d,
		hashNoNonce: common.BytesToHash(h),
		mixDigest:   common.BytesToHash(md),
		//nonce:       n,
		number: 1,
	}

	eth := New()
	nonce, _ := eth.Search(block, nil)
	block.nonce = nonce

	if !eth.Verify(block) {
		t.Error("Block could not be verified")
	}
}
