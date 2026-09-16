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

package hfresh

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/weaviate/sroar"

	"github.com/weaviate/weaviate/adapters/repos/db/helpers"
	"github.com/weaviate/weaviate/adapters/repos/db/vector/hnsw/distancer"
	ent "github.com/weaviate/weaviate/entities/vectorindex/hfresh"
)

// newClaimOrderIndex wires the fixture every claim-order test shares: L2 for
// BOTH the index and the centroid HNSW (the default fixture leaves the
// centroid HNSW on cosine, under which the axis-aligned test vectors used
// here are all near-parallel and indistinguishable). probe == 0 keeps the
// default searchProbe.
func newClaimOrderIndex(t *testing.T, probe uint32) TestHFresh {
	t.Helper()
	uc := ent.NewDefaultUserConfig()
	if probe > 0 {
		uc.SearchProbe = probe
	}
	return newTestIndex(t, uc, nil, withDistanceProvider(distancer.NewL2SquaredProvider()))
}

type postingMember struct {
	id  uint64
	vec []float32
}

// addClaimTestPosting creates one posting holding the given members (in
// order, ids need not be sequential) under the given centroid, and registers
// it with every structure selection consults (centroid HNSW, posting store,
// posting map / sizes).
func addClaimTestPosting(t *testing.T, tf *TestHFresh, centroid []float32, members ...postingMember) {
	t.Helper()

	postingID, posting := createPostingWithVectors(t, tf, [][]float32{members[0].vec}, members[0].id)
	for _, m := range members[1:] {
		tf.Vectors.put(m.id, m.vec)
		require.NoError(t, tf.Index.VersionMap.store.Set(t.Context(), m.id, VectorVersion(1)))
		compressed := tf.Index.quantizer.CompressedBytes(tf.Index.quantizer.Encode(m.vec))
		posting = posting.AddVector(NewVector(m.id, VectorVersion(1), compressed))
	}

	centCompressed := tf.Index.quantizer.CompressedBytes(tf.Index.quantizer.Encode(centroid))
	require.NoError(t, tf.Index.Centroids.Insert(postingID, &Centroid{
		Uncompressed: centroid,
		Compressed:   centCompressed,
	}))
	require.NoError(t, tf.Index.PostingStore.Put(t.Context(), postingID, posting))
	require.NoError(t, tf.Index.setPostingVectorIDs(t.Context(), postingID, posting))
}

// paddedAllowList returns an allowlist passing the given ids, padded with
// nonexistent high ids so Len() clears the flat-search cutoff and the search
// takes the centroid path. The padding ids do not exist in the index, so
// they only affect Len().
func paddedAllowList(ids ...uint64) helpers.AllowList {
	const (
		padLo = uint64(1_000_000)
		padN  = 5_006
	)
	bm := sroar.NewBitmap()
	for _, id := range ids {
		bm.Set(id)
	}
	for i := uint64(0); i < padN; i++ {
		bm.Set(padLo + i)
	}
	return helpers.NewAllowListFromBitmap(bm)
}

// TestFilteredSearchClaimOrder pins a recall bug in the wrapped allowlist:
// the old containsPosting "claimed" a passing vector for the FIRST posting
// probed during centroid search — in the flat-centroid regime that is
// ascending posting-id order, in the graph regime it is traversal order,
// neither of which is query-distance order. A posting whose passing members
// were all claimed by earlier-probed postings reported Contains == false and
// was excluded from centroid selection, even when it was the nearest posting
// to the query. Under probe-budget pressure (searchProbe smaller than the
// number of allowed postings) the claiming posting could miss the selection
// cut while the excluded near posting would have made it — the passing
// vector's postings were then never read and the vector was unreachable for
// that query.
//
// Setup: vector v (the exact nearest neighbor of the query) is replicated in
// two postings — one under a FAR centroid, one under a centroid AT v. Four
// filler postings sit in between; the allowlist passes v and the fillers.
// Posting ids are allocated sequentially, so creation order is claim-probe
// order in the flat-centroid regime: with the far posting first it claims v,
// and pre-fix, under probe pressure, the search returned a filler instead of
// v — while reversing only the posting-id order returned v. The third case
// documents the boundary: with the default probe every allowed posting fits
// the budget, so the claiming posting is always scanned and v is found
// regardless of order.
func TestFilteredSearchClaimOrder(t *testing.T) {
	const (
		vID   = uint64(1000) // the query's exact nearest neighbor
		wBase = uint64(2000) // filler vector ids: wBase+1 ...
		nWs   = 4
	)

	vVec := []float32{0.1, 0.1, 0.1, 0.12}
	farCentroid := []float32{10, 10, 10, 10}
	query := []float32{0.1, 0.1, 0.1, 0.1}

	buildIndex := func(t *testing.T, farPostingFirst bool, probe uint32) TestHFresh {
		tf := newClaimOrderIndex(t, probe)

		vPosting := func() { addClaimTestPosting(t, &tf, farCentroid, postingMember{vID, vVec}) }
		nearPosting := func() { addClaimTestPosting(t, &tf, vVec, postingMember{vID, vVec}) }
		fillers := func() {
			for i := 1; i <= nWs; i++ {
				w := []float32{2, 2, 2, 2 + float32(i)*0.01}
				addClaimTestPosting(t, &tf, w, postingMember{wBase + uint64(i), w})
			}
		}

		if farPostingFirst {
			vPosting()
			fillers()
			nearPosting()
		} else {
			nearPosting()
			fillers()
			vPosting()
		}
		return tf
	}

	cases := []struct {
		name            string
		farPostingFirst bool
		probe           uint32 // 0 = default searchProbe
	}{
		// control: the near posting claims v, the search finds it
		{name: "near posting claims v", farPostingFirst: false, probe: 2},
		// the pinned bug: the far posting claims v under probe pressure —
		// pre-fix this returned a filler; identical data, flipped ids
		{name: "far posting claims v", farPostingFirst: true, probe: 2},
		// boundary: no budget pressure, the claimer is always scanned
		{name: "far posting claims v, no budget pressure", farPostingFirst: true, probe: 0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tf := buildIndex(t, tc.farPostingFirst, tc.probe)

			allowedIDs := []uint64{vID}
			for i := 1; i <= nWs; i++ {
				allowedIDs = append(allowedIDs, wBase+uint64(i))
			}
			allow := paddedAllowList(allowedIDs...)
			defer allow.Close()

			ids, _, err := tf.Index.SearchByVector(t.Context(), query, 1, allow)
			require.NoError(t, err)
			require.Equal(t, []uint64{vID}, ids,
				"the allowlisted exact nearest neighbor must be returned regardless of posting id order")
		})
	}
}

// TestFilteredSearchReplicaDedupe guards the invariant the claim-at-selection
// fix must preserve: a docID replicated across multiple SELECTED postings is
// neither double-scored (one distance computation, via the scan's visited
// dedup) nor double-counted in the result set. Both replica postings get
// selected because each contributes a distinct additional allowed vector.
func TestFilteredSearchReplicaDedupe(t *testing.T) {
	const (
		vID  = uint64(1000)
		w1ID = uint64(2001)
		w2ID = uint64(2002)
	)

	vVec := []float32{0.1, 0.1, 0.1, 0.12}
	w1 := []float32{2, 2, 2, 2}
	// w2 doubles as posting 2's centroid: it must stay within
	// MaxDistanceRatio of the best centroid (~vVec, near-zero distance) or
	// selectCentroids prunes the posting before coverage dedup even runs
	w2 := []float32{0.5, 0.5, 0.5, 0.5}
	query := []float32{0.1, 0.1, 0.1, 0.1}

	tf := newClaimOrderIndex(t, 0)

	// v is replicated in both postings; each posting adds a distinct
	// allowed vector so both survive selection-time coverage dedup
	addClaimTestPosting(t, &tf, vVec, postingMember{vID, vVec}, postingMember{w1ID, w1})
	addClaimTestPosting(t, &tf, w2, postingMember{vID, vVec}, postingMember{w2ID, w2})

	allow := paddedAllowList(vID, w1ID, w2ID)
	defer allow.Close()

	var stats QueryStats
	ids, _, err := tf.Index.SearchByVectorWithStats(t.Context(), query, 3, allow, &stats)
	require.NoError(t, err)

	// no double-counting: every allowed vector exactly once, in exact
	// distance order (v ~0.0004, w2 0.64, w1 14.44)
	require.Equal(t, []uint64{vID, w2ID, w1ID}, ids)

	// both replica postings were selected and scanned
	require.Equal(t, 2, stats.PostingsRead)
	// no double-scoring: v's replica is skipped by the visited dedup, so
	// only 3 distinct members are scanned and scored
	require.Equal(t, 3, stats.MembersScanned)
	require.Equal(t, 3, stats.PassingMembers)
	require.Equal(t, 3, stats.DistanceComps)
}

// TestFilteredSearchSelectionBackfill pins the backfill property of
// selection-time coverage dedup: skipping a redundant posting must free its
// budget slot for the next-ranked posting with new coverage. The centroid
// search list is overfetched (budget x replicas) precisely for this — with a
// list capped at budget, the redundant replica posting between two useful
// ones would burn a slot and the third posting would never be scanned.
func TestFilteredSearchSelectionBackfill(t *testing.T) {
	const (
		vID = uint64(1000)
		wID = uint64(2001)
	)

	vVec := []float32{0.1, 0.1, 0.1, 0.12}
	// replica posting's centroid sits between the v posting and the w
	// posting in rank order, and inside the MaxDistanceRatio bound
	replicaCentroid := []float32{0.11, 0.1, 0.1, 0.1}
	wVec := []float32{0.3, 0.3, 0.3, 0.3}
	query := []float32{0.1, 0.1, 0.1, 0.1}

	tf := newClaimOrderIndex(t, 2)

	addClaimTestPosting(t, &tf, vVec, postingMember{vID, vVec})            // rank 1: covers v
	addClaimTestPosting(t, &tf, replicaCentroid, postingMember{vID, vVec}) // rank 2: redundant replica of v
	addClaimTestPosting(t, &tf, wVec, postingMember{wID, wVec})            // rank 3: covers w, needs backfill

	allow := paddedAllowList(vID, wID)
	defer allow.Close()

	var stats QueryStats
	ids, _, err := tf.Index.SearchByVectorWithStats(t.Context(), query, 2, allow, &stats)
	require.NoError(t, err)
	require.Equal(t, []uint64{vID, wID}, ids,
		"the redundant replica posting must not burn a budget slot: w's posting is scanned via backfill")
	require.Equal(t, 2, stats.PostingsRead)
}
