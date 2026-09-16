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
	"context"
	"iter"

	"github.com/weaviate/weaviate/adapters/repos/db/helpers"
	"github.com/weaviate/weaviate/adapters/repos/db/vector/hnsw/visited"
)

// wrapAllowList bridges a docID allowlist into the posting domain for the
// centroid search: Contains(postingID) answers whether the posting holds at
// least one allowed vector.
//
// The probe-time check is PURE — it never mutates cross-posting state.
// Coverage dedup (each allowed vector justifying only one posting, so the
// probe budget spreads over distinct coverage) happens at selection time via
// selectPosting, in ranked distance order. It used to happen here, at probe
// time, which made results depend on probe order: a far posting explored
// first (posting-id order on the flat path, traversal order under ACORN)
// could claim a vector and get pruned later, while the near posting holding
// the same vector had already been reported disallowed — leaving the vector
// unreachable (see allow_list_claim_order_test.go).
func (h *HFresh) wrapAllowList(ctx context.Context, al helpers.AllowList) *allowList {
	return &allowList{
		AllowList:       al,
		ctx:             ctx,
		h:               h,
		allowedPostings: h.visitedPool.Borrow(),
		claimedVectors:  h.visitedPool.Borrow(),
	}
}

func (h *HFresh) NewAllowListIterator(al helpers.AllowList) helpers.AllowListIterator {
	all := h.PostingMap.Iter()
	next, stop := iter.Pull2(all)

	return &AllowListIterator{
		allowList: al,
		next:      next,
		stop:      stop,
		len:       int(h.Centroids.GetMaxID()),
	}
}

type AllowListIterator struct {
	len       int
	next      func() (uint64, *PostingMetadata, bool)
	stop      func()
	allowList helpers.AllowList
}

func (i *AllowListIterator) Len() int {
	return i.len
}

func (i *AllowListIterator) Stop() {
	i.stop()
}

func (i *AllowListIterator) Next() (uint64, bool) {
	id, metadata, ok := i.next()
	al, isOurType := i.allowList.(*allowList)

	for ok {
		if i.contains(id, metadata, al, isOurType) {
			return id, true
		}
		id, metadata, ok = i.next()
	}

	return id, false
}

func (i *AllowListIterator) contains(id uint64, metadata *PostingMetadata, al *allowList, isOurType bool) bool {
	if isOurType && metadata != nil {
		return al.containsAllowedMember(id, metadata)
	}
	return i.allowList.Contains(id)
}

type allowList struct {
	helpers.AllowList
	ctx context.Context
	h   *HFresh
	// allowedPostings memoizes postings known to contain at least one
	// allowed vector (the answer is stable within a query, so re-probes
	// skip the member scan)
	allowedPostings *visited.SparseSet
	// claimedVectors tracks, at SELECTION time only, which allowed vectors
	// are already covered by a selected posting; selectPosting uses it to
	// skip postings that add no new coverage
	claimedVectors *visited.SparseSet
}

// Contains reports whether the posting holds at least one allowed vector.
// It is pure: probing a posting never affects the answer for another one.
func (a *allowList) Contains(id uint64) bool {
	if a.allowedPostings.Visited(id) {
		return true
	}

	p, err := a.h.PostingMap.Get(a.ctx, id)
	if err != nil {
		return false
	}

	return a.containsAllowedMember(id, p)
}

func (a *allowList) containsAllowedMember(id uint64, p *PostingMetadata) bool {
	if a.allowedPostings.Visited(id) {
		return true
	}

	a.h.PostingMap.RLock(id)
	defer a.h.PostingMap.RUnlock(id)

	for vectorID := range p.Iter() {
		if a.AllowList.Contains(vectorID) {
			a.allowedPostings.Visit(id)
			return true
		}
	}
	return false
}

// selectPosting decides, at final selection time and in ranked distance
// order, whether the posting contributes allowed vectors not already covered
// by a previously selected posting. When it does, all its allowed members
// are claimed (the posting scan will score every one of them) and the
// posting is selected; when every allowed member is already claimed the
// posting is redundant — its allowed members will be scanned via the
// posting(s) that claimed them — and skipping it saves the read.
func (a *allowList) selectPosting(id uint64) bool {
	p, err := a.h.PostingMap.Get(a.ctx, id)
	if err != nil {
		return false
	}

	a.h.PostingMap.RLock(id)
	defer a.h.PostingMap.RUnlock(id)

	newCoverage := false
	for vectorID := range p.Iter() {
		if a.AllowList.Contains(vectorID) && !a.claimedVectors.Visited(vectorID) {
			a.claimedVectors.Visit(vectorID)
			newCoverage = true
		}
	}
	return newCoverage
}

// resetClaims discards all selection-time claims so a fresh selection pass
// can run against a re-searched, deeper centroid list (lazy deepening).
// Probe-time memoization (allowedPostings) stays valid: it records a pure
// property of the posting.
func (a *allowList) resetClaims() {
	a.h.visitedPool.Return(a.claimedVectors)
	a.claimedVectors = a.h.visitedPool.Borrow()
}

// Iterator implements [helpers.AllowList].
func (a *allowList) Iterator() helpers.AllowListIterator {
	return a.h.NewAllowListIterator(a)
}

// Len implements [helpers.AllowList].
func (a *allowList) Len() int {
	return int(a.h.Centroids.GetMaxID())
}

func (a *allowList) Close() {
	a.h.visitedPool.Return(a.allowedPostings)
	a.h.visitedPool.Return(a.claimedVectors)
}
