/** @file clustering.h
 * @brief Xapian::Clustering API class
 */
/* Copyright (C) 2014 Andrei Sfrent
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
 */

#ifndef XAPIAN_INCLUDED_CLUSTERING_H
#define XAPIAN_INCLUDED_CLUSTERING_H

#include <xapian/enquire.h>
#include <xapian/visibility.h>

#include <map>
#include <set>
#include <vector>
#include <cmath>

namespace Xapian {

// INTERNAL USE ONLY.
// Feature vector, a mapping from coordinate to double values.
template<typename T>
class FeatureVector : public std::map<T, double> {
  private:
    double _norm;
    bool _cached;

    void compute_norm() {
      _norm = 0.0;
      for (typename std::map<T, double>::const_iterator it = this->begin();
          it != this->end(); ++it) {
        const double& value = it->second;
        _norm += value * value;
      }
      if (_norm != 0.0) {
        _norm = sqrt(_norm);
      }
      _cached = true;
    }

  public:
    FeatureVector() {
      mark_dirty();
    }

    double norm() {
      if (!_cached) {
        compute_norm();
      }
      return _norm;
    }

    bool is_zero() {
      return norm() == 0.0;
    }

    void mark_dirty() {
      _cached = false;
    }
};

// INTERNAL USE ONLY. Only the type should be visible to the user so it
// can specify it to the clusterer.
// Similarity metric: cosine similarity.
class CosineSimilarity {
  private:
    template<typename T>
    double compute_inner_product(
        const FeatureVector<T> *fv1, const FeatureVector<T> *fv2) {
      double inner = 0.0;

      typename std::map<T, double>::const_iterator it1 = fv1->begin();
      typename std::map<T, double>::const_iterator it2 = fv2->begin();
      while (it1 != fv1->end() && it2 != fv2->end()) {
        if (it1->first == it2->first) {
          inner += it1->second * it2->second;
          ++it1, ++it2;
        } else if (it1->first < it2->first) {
          ++it1;
        } else {
          ++it2;
        }
      }

      return inner;
    }
  public:
    // Computes the similarity between two FeatureVectors of regardless of
    // the coordinate type.
    template<typename T>
    double similarity(const FeatureVector<T> *fv1, const FeatureVector<T> *fv2) {
      double norm1 = fv1->norm();
      double norm2 = fv2->norm();
      double inner_product = compute_inner_product(fv1, fv2);
      return inner_product / (norm1 * norm2);
    }
};

// // INTERNAL USE ONLY.
// // Builds the initial feature vectors from a collection of documents.
// class FeatureVectorsBuilder {
//   public:
//     // Some feature vectors can't be constructed until all documents have been
//     // analyzed. This method should be called in order for the builder to
//     // analyze the documents. The method should always be called by the vector
//     // space factory methods.
//     virtual void analyze_documents(const Xapian::MSet& mset) = 0;
//
//     // This method uses the information gathered in present documents in order
//     // to actually build the feature vectors. The caller should take care of the
//     // allocated memory.
//     virtual std::map<docid, FeatureVector<std::string>*> compute_feature_vectors() = 0;
// };
//
// // INTERNAL USE ONLY. Only the type should be visible to the user so it
// // can specify it to the clusterer.
// // Implementation of FeatureVectorBuilder that computes tf-idf for each word in
// // the documents.
// class TfidfBuilder : public FeatureVectorsBuilder {
//   public:
//     // This function will analyze the contents and compute needed statistics.
//     void analyze_documents(const Xapian::MSet& mset);
//
//     // Computes the final vectors from the statistics gathered by present_documents.
//     void sed::map<docid, FeatureVector<std::string>*> compute_feature_vectors();
// };
//
// // INTERNAL USE ONLY
// // This class keeps track of the feature vectors and is the main link between
// // the clustering algorithm and mathematics. It is the only one that should be
// // aware of the way feature vectors are built and the similarity metric. This
// // could be changed in the future for performance reasons.
// // Coordinates are transformed from theier original type to int for faster
// // access. The clustering algorithm will work on integer indexes.
// template<typename FeatureVectorsBuilder, typename SimilarityMetric>
// class VectorSpace {
//   private:
//     // Maps the new coordinates back into the old ones so we can access the
//     // actual terms at a later point if we need.
//     std::map<int, std::string> _coordinate_mapping;
//
//     // Maps the index of a datapoint in the vector space to its original index
//     // (docid). The information is needed in order to present the clusters to
//     // the user in terms of docids.
//     std::map<int, docid> _datapoint_to_object;
//
//     // Similarity metric used in this vector space.
//     SimilarityMetric _similarity;
//
//   public:
//     // Data points (each document will have a different index in this vector).
//     std::vector<FeatureVector<int>*> data;
//
//     // Computes similarity between two indexes in the datapoints.
//     double similarity(int index1, int index2);
//
//     // Computes similarity between two given feature vectors.
//     double similarity(const FeatureVector<int>* fv1, const FeatureVector<int>* fv2);
//
//     // Computes similarity between a given feature vector and the feature vector
//     // at index in the datapoints.
//     double similarity(const FeatureVector<int>* fv, int index);
//
//     // Builds a VectorSpace object for use in the clusterer.
//     static VectorSpace *fromMSet(const Xapian::MSet& mset);
// };

// VISIBLE TO THE USER.
// A cluster is a set of documents. Objects will be constructed by the
// clustering algorithm and will be immutable for the user.
class XAPIAN_VISIBILITY_DEFAULT Cluster {
  // friend class KMeansClusterer; // Although is should be any type of clusterer.

  private:
    // Cluster ID.
    int _id;

    // Contents of the cluster.
    std::set<docid> _contents;

  public:
    const std::set<docid>& get_contents() const {
      return _contents;
    }

    bool contains(docid doc) const {
      return _contents.find(doc) != _contents.end();
    }

    size_t size() const {
      return _contents.size();
    }

    int id() const {
      return _id;
    }
};

// VISIBLE TO THE USER.
// A collection of clusters, the result of a clustering algorithm. Objects will
// be constructed by the clustering algorithm and will be immutable for the
// user.
class XAPIAN_VISIBILITY_DEFAULT Clusters {
  // friend class KMeansClusterer; // Although is should be any type of clusterer.

  private:
    // Maps documents to clusters.
    std::map<docid, Cluster> _doc_to_cluster;

    // Set of all clusters.
    std::set<Cluster> _clusters;

  public:
    size_t count() const {
      return _clusters.size();
    }

    const Cluster& get(docid doc) const {
      std::map<docid, Cluster>::const_iterator it =
          _doc_to_cluster.find(doc);
      if (it == _doc_to_cluster.end()) {
        // FIXME error.
      }
      return it->second;
    }

    const std::set<Cluster>& get_all() const {
      return _clusters;
    }
};

// // VISIBLE TO THE USER.
// template<typename FeatureVectorBuilder, typename SimilarityMetric>
// class XAPIAN_VISIBILITY_DEFAULT KMeansClusterer {
//   private:
//     // Vector space for use in the clusterer.
//     VectorSpace<FeatureVectorsBuilder, SimilarityMetric> *_vector_space;
//
//   public:
//     // Method that actually clusters the data.
//     void cluster(const Xapian::MSet& mset, int cluster_count);
//
//     // Returns the results of the clustering.
//     const Clusters& get_results();
// };
}

#endif  // XAPIAN_INCLUDED_CLUSTERING_H
