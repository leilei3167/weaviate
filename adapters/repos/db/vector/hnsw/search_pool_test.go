//                           _       _
// __      _____  __ ___   ___  __ _| |_ ___
// \ \ /\ / / _ \/ _` \ \ / / |/ _` | __/ _ \
//  \ V  V /  __/ (_| |\ V /| | (_| | ||  __/
//   \_/\_/ \___|\__,_| \_/ |_|\__,_|\__\___|
//
//  Copyright © 2016 - 2026 Weaviate B.V. All rights reserved.
//
//  CONTACT: hello@weaviate.io
//

package hnsw

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/weaviate/weaviate/adapters/repos/db/helpers"
	"github.com/weaviate/weaviate/adapters/repos/db/lsmkv"
	"github.com/weaviate/weaviate/adapters/repos/db/vector/common"
	"github.com/weaviate/weaviate/adapters/repos/db/vector/hnsw/distancer"
	"github.com/weaviate/weaviate/adapters/repos/db/vector/testinghelpers"
	"github.com/weaviate/weaviate/entities/cyclemanager"
	ent "github.com/weaviate/weaviate/entities/vectorindex/hnsw"
	"github.com/weaviate/weaviate/usecases/memwatch"
)

// newPoolTestIndex builds and seeds an index for the pool-search tests.
// AcornFilterRatio is pinned high — mirroring the HFresh centroid config,
// the pool search's consumer — because the strategy-selection ratio check
// samples the entrypoint's neighborhood, and HNSW level assignment is
// randomly seeded per run: on an unlucky graph a normal ratio silently
// demotes ACORN to RRE, where capture never arms and the beyond-ef and
// CaptureActive assertions fail (this exact flake hit CI).
// MakeBucketOptions serves the compressed variants.
func newPoolTestIndex(t *testing.T, uc ent.UserConfig, vectors [][]float32) *hnsw {
	t.Helper()
	ctx := context.Background()

	store := testinghelpers.NewDummyStore(t)
	t.Cleanup(func() { store.Shutdown(ctx) })

	index, err := New(Config{
		RootPath:              t.TempDir(),
		ID:                    "pool-test",
		MakeCommitLoggerThunk: MakeNoopCommitLogger,
		DistanceProvider:      distancer.NewCosineDistanceProvider(),
		AllocChecker:          memwatch.NewDummyMonitor(),
		MakeBucketOptions:     lsmkv.MakeNoopBucketOptions,
		AcornFilterRatio:      1000,
		VectorForIDThunk: func(ctx context.Context, id uint64) ([]float32, error) {
			return vectors[int(id)], nil
		},
		GetViewThunk:                 func() common.BucketView { return &noopBucketView{} },
		TempVectorForIDWithViewThunk: TempVectorForIDWithViewThunk(vectors),
	}, uc, cyclemanager.NewCallbackGroupNoop(), store)
	require.NoError(t, err)
	t.Cleanup(func() { index.Shutdown(ctx) })

	for id := uint64(0); id < uint64(len(vectors)); id++ {
		require.NoError(t, index.Add(ctx, id, vectors[id]))
	}
	return index
}

// strideAllowList allows every (n/allowed)-th id of an n-vector corpus.
func strideAllowList(n, allowed int) ([]uint64, helpers.AllowList) {
	ids := make([]uint64, 0, allowed)
	for id := uint64(0); id < uint64(n); id += uint64(n / allowed) {
		ids = append(ids, id)
	}
	return ids, helpers.NewAllowList(ids...)
}

// TestSearchByVectorWithPoolFlatRescore pins the flat-path contract of
// SearchByVectorWithPool on a rescoring compressed index: flatSearch caps
// its pre-rescore heap at the limit parameter whenever shouldRescore() is
// true, so the pool search must pass a poolK-sized limit (not just
// searchTimeEF(k)) or the returned list silently truncates below poolK.
// BQ is used because it activates compression at construction with
// rescoring enabled, no training pass required. This is the common regime
// for HFresh centroid indexes below the 40k flat cutoff (any corpus under
// ~2.5M vectors), where the deeper list must reach full poolK depth.
func TestSearchByVectorWithPoolFlatRescore(t *testing.T) {
	const (
		n       = 300
		allowed = 150
		k       = 5
		ef      = 32
		poolK   = 100
	)

	vectors, queries := testinghelpers.RandomVecsFixedSeed(n, 1, 8)

	uc := ent.UserConfig{
		MaxConnections:        16,
		EFConstruction:        32,
		EF:                    ef,
		VectorCacheMaxObjects: 100000,
		// literal construction skips SetDefaults, so the cutoff must be set
		// explicitly or it is 0 and the flat branch never triggers
		FlatSearchCutoff: 40000,
	}
	uc.BQ.Enabled = true // compressed from construction, shouldRescore() true

	index := newPoolTestIndex(t, uc, vectors)

	// 150 allowed of 300 stays under the flat cutoff, so the pool search
	// takes the flat branch
	_, allow := strideAllowList(n, allowed)
	defer allow.Close()

	ids, dists, stats, err := index.SearchByVectorWithPool(context.Background(), queries[0], k, poolK, allow)
	require.NoError(t, err)

	// the whole point: the rescore-capped heap must not truncate the pool
	// below poolK (pre-fix this returned at most searchTimeEF(k)=32)
	require.Len(t, ids, poolK)
	require.Equal(t, poolK, stats.PoolSize)
	// the flat path is exhaustive up to poolK, so its depth is conclusive
	require.True(t, stats.CaptureActive)

	for i := 1; i < len(dists); i++ {
		require.GreaterOrEqual(t, dists[i], dists[i-1], "flat pool must be sorted by distance")
	}
}

// TestSearchByVectorWithPoolEntrypointSeedDuplicate reproduces the
// duplicate-entrypoint hazard: on the ACORN path the allow-list seed loop
// can enqueue the global entrypoint a second time (the descent already
// contributed it), and insertViableEntrypointsAsCandidatesAndResults must
// not insert both occurrences into the result heap — one copy would stay in
// the beam while the other is popped into the discard pool, and the merge
// would return the same id twice while burning a poolK slot. Two
// ingredients make the double-seeding deterministic regardless of which
// node the (randomly leveled) graph elected as entrypoint: the query IS the
// entrypoint's own vector, so the per-level descent can never improve on it
// (distance 0) and the global entrypoint survives to level 0; and the
// allowlist is exactly {entrypoint}, so the seed loop re-enqueues it.
func TestSearchByVectorWithPoolEntrypointSeedDuplicate(t *testing.T) {
	const (
		n     = 300
		k     = 2
		ef    = 8
		poolK = 10
	)

	ctx := context.Background()
	vectors, _ := testinghelpers.RandomVecsFixedSeed(n, 1, 8)

	index := newPoolTestIndex(t, ent.UserConfig{
		MaxConnections:        16,
		EFConstruction:        32,
		EF:                    ef,
		VectorCacheMaxObjects: 100000,
		FilterStrategy:        ent.FilterStrategyAcorn,
		FlatSearchCutoff:      1, // force the graph path
	}, vectors)

	ep := index.entryPointID
	query := vectors[ep]
	allow := helpers.NewAllowList(ep)
	defer allow.Close()

	poolIDs, _, stats, err := index.SearchByVectorWithPool(ctx, query, k, poolK, allow)
	require.NoError(t, err)
	require.Equal(t, []uint64{ep}, poolIDs,
		"the doubly-seeded entrypoint must appear exactly once in the pooled output")
	require.Equal(t, 1, stats.PoolSize)

	// the plain path shares the entrypoint seeding and must dedup too
	plainIDs, _, err := index.SearchByVector(ctx, query, k, allow)
	require.NoError(t, err)
	require.Equal(t, []uint64{ep}, plainIDs,
		"the plain search must not return the doubly-seeded entrypoint twice")
}

// TestSearchByVectorWithPool pins the pool-reuse contract on the ACORN path:
// the deeper ranked list is served from candidates the k-search already
// evaluated, WITHOUT touching the beam or its termination. Two properties
// follow and are asserted here:
//
//  1. the first k pooled results are identical to a plain k-search (capture
//     must not perturb the traversal), and
//  2. the pool reaches beyond ef (the plain search could never return more
//     than ef results), while staying sorted, deduplicated, and allowed.
func TestSearchByVectorWithPool(t *testing.T) {
	const (
		n       = 1000
		allowed = 200
		k       = 5
		// ef below the 10 allowlist seeds the ACORN path inserts into the
		// result heap makes beyond-ef depth a STRUCTURAL guarantee: how far
		// the beam explores past ef varies per run (level assignment is
		// randomly seeded, so every run builds a different graph) and per
		// platform, so the assertion must not depend on exploration volume.
		ef    = 8
		poolK = 50
	)

	ctx := context.Background()
	vectors, queries := testinghelpers.RandomVecsFixedSeed(n, 1, 8)

	index := newPoolTestIndex(t, ent.UserConfig{
		MaxConnections:        16,
		EFConstruction:        32,
		EF:                    ef,
		VectorCacheMaxObjects: 100000,
		FilterStrategy:        ent.FilterStrategyAcorn,
		// force the graph path: the default cutoff (40k) would send an
		// allowlist this small to flat search
		FlatSearchCutoff: 1,
	}, vectors)

	allowedIDs, allow := strideAllowList(n, allowed)
	defer allow.Close()
	allowSet := make(map[uint64]bool, len(allowedIDs))
	for _, id := range allowedIDs {
		allowSet[id] = true
	}

	query := queries[0]

	plainIDs, plainDists, err := index.SearchByVector(ctx, query, k, allow)
	require.NoError(t, err)
	require.Len(t, plainIDs, k)

	poolIDs, poolDists, stats, err := index.SearchByVectorWithPool(ctx, query, k, poolK, allow)
	require.NoError(t, err)

	// (2) beyond-ef depth: the plain search caps at ef results, the pool
	// must reach past that from the already-evaluated candidates
	require.Greater(t, len(poolIDs), ef, "pool must be deeper than the beam")
	require.LessOrEqual(t, len(poolIDs), poolK)
	require.Equal(t, len(poolIDs), stats.PoolSize)
	require.Positive(t, stats.DistanceComps)
	require.True(t, stats.CaptureActive)

	// (1) capture must not perturb the traversal: identical top-k
	require.Equal(t, plainIDs, poolIDs[:k])
	require.Equal(t, plainDists, poolDists[:k])

	// ranked, deduplicated, allowed
	seen := make(map[uint64]bool, len(poolIDs))
	for i, id := range poolIDs {
		require.True(t, allowSet[id], "pooled id %d not in the allowlist", id)
		require.False(t, seen[id], "pooled id %d duplicated", id)
		seen[id] = true
		if i > 0 {
			require.GreaterOrEqual(t, poolDists[i], poolDists[i-1], "pool must be sorted by distance")
		}
	}
}
