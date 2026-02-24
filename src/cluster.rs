use itertools::Itertools;
use ndarray::{Array2, Axis};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use hdbscan::{DistanceMetric, Hdbscan, HdbscanHyperParams};
use crate::convert_with_params;
use anyhow::Result;

#[derive(Debug, Clone, Copy)]
pub struct PahcConfig {
    pub merge_cutoff: f32,
    pub min_cluster_size: usize,
    pub absorb_cutoff: f32,
    pub max_speakers: Option<usize>,
}

impl Default for PahcConfig {
    fn default() -> Self {
        Self {
            merge_cutoff: 0.3,
            min_cluster_size: 3,
            absorb_cutoff: 0.0,
            max_speakers: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Pahc {
    cfg: PahcConfig,
}

impl Pahc {
    pub fn new(cfg: PahcConfig) -> Self {
        Self { cfg }
    }

    pub fn fit_predict(&self, labels: &[i32], embeddings: &Array2<f32>) -> Vec<usize> {
        assert_eq!(
            labels.len(),
            embeddings.len_of(Axis(0)),
            "labels and embeddings length mismatch"
        );

        let mut state = PahcState::new(self.cfg, labels, embeddings);
        state.merge_cluster();
        state.absorb_cluster();

        if let Some(k) = self.cfg.max_speakers {
            state.limit_clusters(k);
        }

        state.relabel_cluster()
    }
}

#[derive(Debug, Clone, Copy)]
struct HeapItem {
    score: f32,
    i: usize,
    j: usize,
}

impl PartialEq for HeapItem {
    fn eq(&self, o: &Self) -> bool {
        self.score == o.score && self.i == o.i && self.j == o.j
    }
}
impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal)
    }
}

#[derive(Clone)]
struct PahcState<'a> {
    cfg: PahcConfig,
    labels: &'a [i32],
    embeddings: &'a Array2<f32>,

    active_clusters: HashSet<usize>,
    label_map: HashMap<usize, Vec<usize>>,
    cost_map: HashMap<(usize, usize), f32>,
    heap: BinaryHeap<HeapItem>,

    next_index: usize,
    num_labeled: usize,
}

impl<'a> PahcState<'a> {
    fn new(cfg: PahcConfig, labels: &'a [i32], embeddings: &'a Array2<f32>) -> Self {
        let mut s = Self {
            cfg,
            labels,
            embeddings,
            active_clusters: HashSet::new(),
            label_map: HashMap::new(),
            cost_map: HashMap::new(),
            heap: BinaryHeap::new(),
            next_index: 0,
            num_labeled: 0,
        };
        s.build_label_map();
        s.build_cost_map();
        s
    }

    fn build_label_map(&mut self) {
        let mut tmp: HashMap<i32, Vec<usize>> = HashMap::new();

        for (idx, &lbl) in self.labels.iter().enumerate() {
            tmp.entry(lbl).or_default().push(idx);
        }

        let mut seen: Vec<i32> = tmp.keys().cloned().collect();
        seen.sort_unstable();

        let has_noise = seen.contains(&-1);
        self.num_labeled = if has_noise { seen.len() - 1 } else { seen.len() };

        let mut label_map = HashMap::new();
        let mut cid = 0;

        for lbl in seen.iter().filter(|&&x| x != -1) {
            let Some(v) = tmp.remove(lbl) else { continue };
            label_map.insert(cid, v);
            cid += 1;
        }

        if let Some(noise) = tmp.remove(&-1) {
            for idx in noise {
                label_map.insert(cid, vec![idx]);
                cid += 1;
            }
        }

        self.label_map = label_map;
        self.next_index = cid;
    }

    fn build_cost_map(&mut self) {
        self.active_clusters = (0..self.next_index).collect();

        let ids: Vec<usize> = self.label_map.keys().cloned().collect();

        for (a_pos, &i) in ids.iter().enumerate() {
            for &j in ids.iter().skip(a_pos + 1) {
                let key = (i.min(j), i.max(j));

                let Some(i_idx) = self.label_map.get(&i) else { continue };
                let Some(j_idx) = self.label_map.get(&j) else { continue };

                if i < self.num_labeled && j < self.num_labeled {
                    self.cost_map.insert(key, f32::NEG_INFINITY);
                    continue;
                }

                let cost = self.compute_cost(i_idx, j_idx);
                self.cost_map.insert(key, cost);

                let factor = (i_idx.len() * j_idx.len()) as f32;
                let norm = if factor > 0.0 { cost / factor } else { f32::NEG_INFINITY };

                if norm >= self.cfg.merge_cutoff {
                    self.heap.push(HeapItem { score: norm, i, j });
                }
            }
        }
    }

    fn merge_cluster(&mut self) {
        while let Some(HeapItem { i, j, .. }) = self.heap.pop() {
            if self.active_clusters.contains(&i) && self.active_clusters.contains(&j) {
                self.merge(i, j);
            }
        }
    }

    fn absorb_cluster(&mut self) {
        let (minor, major): (HashSet<_>, HashSet<_>) = self.label_map.iter().map(|(&cid, v)| (cid, v.len())).fold(
            (HashSet::new(), HashSet::new()),
            |(mut mi, mut ma), (cid, sz)| {
                if sz < self.cfg.min_cluster_size {
                    mi.insert(cid);
                } else {
                    ma.insert(cid);
                }
                (mi, ma)
            },
        );

        if major.is_empty() {
            return;
        }

        for i in minor {
            let i_idx = match self.label_map.get(&i) {
                Some(v) => v,
                None => continue,
            };

            let mut best = (f32::NEG_INFINITY, None);

            for &j in &major {
                if i == j {
                    continue;
                }

                let Some(j_idx) = self.label_map.get(&j) else { continue };
                let key = (i.min(j), i.max(j));
                let cost = *self.cost_map.get(&key).unwrap_or(&f32::NEG_INFINITY);

                let factor = (i_idx.len() * j_idx.len()) as f32;
                let norm = if factor > 0.0 { cost / factor } else { f32::NEG_INFINITY };

                if norm > best.0 {
                    best = (norm, Some(j));
                }
            }

            if best.0 >= self.cfg.absorb_cutoff
                && let Some(target) = best.1
                && let Some(mut v) = self.label_map.remove(&i)
                && let Some(dst) = self.label_map.get_mut(&target)
            {
                dst.append(&mut v);
                self.eliminate(i);
            }
        }
    }

    fn limit_clusters(&mut self, k: usize) {
        if self.label_map.len() <= k {
            return;
        }

        let mut clusters: Vec<(usize, usize)> = self.label_map.iter().map(|(&cid, v)| (cid, v.len())).collect();

        clusters.sort_by(|a, b| b.1.cmp(&a.1));

        let major: Vec<usize> = clusters.iter().take(k).map(|(cid, _)| *cid).collect();
        let minor: Vec<usize> = clusters.iter().skip(k).map(|(cid, _)| *cid).collect();

        for i in minor {
            let i_idx = match self.label_map.get(&i) {
                Some(v) => v.clone(),
                None => continue,
            };

            let mut best = (f32::NEG_INFINITY, None);

            for &j in &major {
                let Some(j_idx) = self.label_map.get(&j) else { continue };
                let cost = self.compute_cost(&i_idx, j_idx);

                let factor = (i_idx.len() * j_idx.len()) as f32;
                let norm = if factor > 0.0 { cost / factor } else { f32::NEG_INFINITY };

                if norm > best.0 {
                    best = (norm, Some(j));
                }
            }

            if let Some(target) = best.1
                && let Some(mut v) = self.label_map.remove(&i)
                && let Some(dst) = self.label_map.get_mut(&target)
            {
                dst.append(&mut v);
                self.eliminate(i);
            }
        }
    }

    fn relabel_cluster(&self) -> Vec<usize> {
        let mut result = vec![usize::MAX; self.labels.len()];

        for (&cid, idxs) in &self.label_map {
            for &i in idxs {
                result[i] = cid;
            }
        }

        let mut map = HashMap::new();
        let mut next = 0usize;

        for v in result.iter_mut() {
            let entry = map.entry(*v).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            });
            *v = *entry;
        }

        result
    }

    fn compute_cost(&self, a: &[usize], b: &[usize]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return f32::NEG_INFINITY;
        }

        let dim = self.embeddings.len_of(Axis(1));

        let sum_norm = |inds: &[usize]| -> Vec<f32> {
            let mut acc = vec![0.0; dim];
            for &idx in inds {
                let row_view = self.embeddings.row(idx);
                let vn_vec;
                let row_slice: &[f32] = if let Some(s) = row_view.to_slice() {
                    s
                } else {
                    vn_vec = row_view.to_vec();
                    &vn_vec
                };
                let vn = l2norm(row_slice);
                for d in 0..dim {
                    acc[d] += vn[d];
                }
            }
            acc
        };

        let a_sum = sum_norm(a);
        let b_sum = sum_norm(b);
        dot(&a_sum, &b_sum)
    }

    fn eliminate(&mut self, cid: usize) {
        self.label_map.remove(&cid);
        self.active_clusters.remove(&cid);
    }

    fn merge(&mut self, i: usize, j: usize) {
        let a = match self.label_map.get(&i) {
            Some(v) => v.clone(),
            None => return,
        };
        let b = match self.label_map.get(&j) {
            Some(v) => v.clone(),
            None => return,
        };

        let new_id = self.next_index;
        let others: Vec<usize> = self.label_map.keys().cloned().filter(|&x| x != i && x != j).collect();

        for k in others {
            let Some(k_idx) = self.label_map.get(&k) else { continue };

            let c1 = self.cost_map.get(&(i.min(k), i.max(k))).cloned().unwrap_or(0.0);
            let c2 = self.cost_map.get(&(j.min(k), j.max(k))).cloned().unwrap_or(0.0);

            let cost = c1 + c2;
            self.cost_map.insert((k.min(new_id), k.max(new_id)), cost);

            let new_size = (a.len() + b.len()) as f32;
            let factor = new_size * k_idx.len() as f32;
            let norm = if factor > 0.0 { cost / factor } else { f32::NEG_INFINITY };

            if norm >= self.cfg.merge_cutoff {
                self.heap.push(HeapItem {
                    score: norm,
                    i: k,
                    j: new_id,
                });
            }
        }

        let mut merged = a;
        merged.extend(&b);

        self.label_map.insert(new_id, merged);
        self.active_clusters.insert(new_id);

        self.eliminate(i);
        self.eliminate(j);

        self.next_index += 1;
    }
}

fn l2norm(x: &[f32]) -> Vec<f32> {
    let n = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    if n == 0.0 {
        return x.to_vec();
    }
    x.iter().map(|v| v / n).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip_eq(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn cluster(embeddings: &Array2<f32>, max_speakers: Option<usize>) -> Result<Vec<usize>, Box<dyn std::error::Error + Send + Sync>> {
    let n = embeddings.len_of(Axis(0));
    if n == 0 {
        return Ok(vec![]);
    }
    if n <= 2 {
        return Ok(vec![0; n]);
    }

    let num_components = std::cmp::min(32, embeddings.len_of(Axis(1)));
    let embeddings_f64: Array2<f64> = embeddings.mapv(|x| x as f64);

    let umap_embeddings = convert_with_params(
        embeddings_f64.view(),
        num_components,
        16,
        0.05,
        Some(2023),
        true,
    )?;

    let umap_vec: Vec<Vec<f64>> = umap_embeddings
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();

    let hyper_params = HdbscanHyperParams::builder()
        .min_cluster_size(4)
        .dist_metric(DistanceMetric::Euclidean)
        .build();

    let clusterer = Hdbscan::new(&umap_vec, hyper_params);
    let mut labels = clusterer.cluster().expect("HDBSCAN failed");

    if labels.iter().all(|&l| l < 0) {
        labels.iter_mut().for_each(|l| *l = 0);
    }

    let cfg = PahcConfig {
        merge_cutoff: 0.3,
        min_cluster_size: 3,
        absorb_cutoff: 0.0,
        max_speakers,
    };

    let pahc = Pahc::new(cfg);
    Ok(pahc.fit_predict(&labels, embeddings))
}
