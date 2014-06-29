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
#include <xapian/document.h>

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

// INTERNAL USE ONLY. Only the type should be visible to the user so it
// can specify it to the clusterer.
// Implementation of FeatureVectorBuilder that computes tf-idf for each word in
// the documents.
class TfidfBuilder {
  private:
    double doc_count;
    std::map<std::string, int> _idf_counts;
    std::map<docid, std::map<std::string, int> > _tf_counts;
    std::vector<docid> _docs;

    void update_tf_statistics(docid doc, const std::string& term) {
      // We use the boolean frequencies tf.
      _tf_counts[doc][term] = 1;
    }

    void update_idf_statistics(const std::string& term) {
      ++_idf_counts[term];
    }

    void gather_statistics(const Xapian::MSet& mset) {
      // Store the number of documents (double for performance).
      doc_count = static_cast<double>(mset.size());

      // For every document in the MSet.
      for (MSetIterator mset_it = mset.begin();
          mset_it != mset.end(); ++mset_it) {
        Document current_doc = mset_it.get_document();
        docid current_docid = current_doc.get_docid();

        // Store the docid.
        _docs.push_back(current_docid);

        // For every term in the current document.
        for (TermIterator term_it = current_doc.termlist_begin();
            term_it != current_doc.termlist_end(); ++term_it) {
          const std::string& current_term = *term_it;
          // Update needed statistics.
          update_tf_statistics(current_docid, current_term);
          update_idf_statistics(current_term);
        }
      }
    }

    void update_feature_vectors(
        std::map<docid, FeatureVector<std::string>*>* feature_vectors) {
      feature_vectors->clear();

      // First, we compute the idf value for each term.
      std::map<std::string, double> idf;
      for (std::map<std::string, int>::const_iterator it = _idf_counts.begin();
          it != _idf_counts.end(); ++it) {
        const std::string& term = it->first;
        double term_occurrences = static_cast<double>(it->second);
        idf[term] = log(doc_count / term_occurrences);
      }

      // We now compute feature vectors by multiplying tf and idf.
      for (std::vector<docid>::const_iterator it = _docs.begin();
          it != _docs.end(); ++it) {
        // Allocate new feature vector.
        FeatureVector<std::string> *fv = new FeatureVector<std::string>();

        const std::map<std::string, int>& tf_for_current_doc = _tf_counts[*it];

        // Add all the terms and their tfidf to the feature vector.
        std::map<std::string, int>::const_iterator term_it;
        for (term_it = tf_for_current_doc.begin();
            term_it != tf_for_current_doc.end(); ++term_it) {
          const std::string& term = term_it->first;

          // Get current tf and idf.
          double current_tf = static_cast<double>(term_it->second);
          double current_idf = idf[term];

          // Multiply them together and store the result.
          double tfidf = current_tf * current_idf;
          fv->insert(std::make_pair(term, tfidf));
        }

        // Add the feature vector to the collection, keyed by its corresponding
        // docid.
        feature_vectors->insert(std::make_pair(*it, fv));
      }
    }

  public:
    void compute_feature_vectors(
        const Xapian::MSet& mset,
        std::map<docid, FeatureVector<std::string>*>* feature_vectors) {
      gather_statistics(mset);
      update_feature_vectors(feature_vectors);
    }
};

// INTERNAL USE ONLY
// This class keeps track of the feature vectors and is the main link between
// the clustering algorithm and mathematics. It is the only one that should be
// aware of the way feature vectors are built and the similarity metric. This
// could be changed in the future for performance reasons.
// Coordinates are transformed from theier original type to int for faster
// access. The clustering algorithm will work on integer indexes.
template<typename FeatureVectorsBuilder, typename SimilarityMetric>
class VectorSpace {
  private:
    // Maps the new coordinates back into the old ones so we can access the
    // actual terms at a later point if we need.
    std::map<int, std::string> _coordinate_mapping;

    // Maps the index of a datapoint in the vector space to its original index
    // (docid). The information is needed in order to present the clusters to
    // the user in terms of docids.
    std::map<int, docid> _datapoint_to_object;

    // Similarity metric used in this vector space.
    SimilarityMetric _similarity_metric;

  public:
    // Data points (each document will have a different index in this vector).
    std::vector<FeatureVector<int>*> data;

    // Computes similarity between two indexes in the datapoints.
    double similarity(int index1, int index2) {
      // TODO add assertions for indexes.
      return similarity(data[index1], data[index2]);
    }

    // Computes similarity between two given feature vectors.
    double similarity(const FeatureVector<int>* fv1, const FeatureVector<int>* fv2) {
      return _similarity_metric.similarity(fv1, fv2);
    }

    // Computes similarity between a given feature vector and the feature vector
    // at index in the datapoints.
    double similarity(const FeatureVector<int>* fv, int index) {
      return similarity(fv, data[index]);
    }

    // Builds a VectorSpace object for use in the clusterer.
    static VectorSpace *fromMSet(const Xapian::MSet& mset) {
      // Use the feature vector builder to get string indexed feature vectors.
      FeatureVectorsBuilder feature_vector_builder;
      std::map<docid, FeatureVector<std::string>*> original_vectors;
      feature_vector_builder.compute_feature_vectors(mset, &original_vectors);

      // Allocate new vector space.
      VectorSpace *vector_space = new VectorSpace;

      // Iterate through original vectors, fill info in vector space and free them.
      std::map<std::string, int> term_to_index;
      std::map<docid, FeatureVector<std::string>*>::const_iterator it;
      for (it = original_vectors.begin(); it != original_vectors.end(); ++it) {
        docid current_doc = it->first;
        const FeatureVector<std::string>& original_fv = *(it->second);
        FeatureVector<int> *reindexed_fv = new FeatureVector<int>;
        FeatureVector<std::string>::const_iterator fvit;
        for (fvit = original_fv.begin(); fvit != original_fv.end(); ++fvit) {
          if(term_to_index.find(fvit->first) == term_to_index.end()) {
            term_to_index[fvit->first] = static_cast<int>(term_to_index.size());
          }
          int new_term_index = term_to_index[fvit->first];
          reindexed_fv->insert(std::make_pair(new_term_index, fvit->second));
        }
        delete it->second;

        // Update vector space.
        int reindexed_fv_index = static_cast<int>(vector_space->data.size());
        vector_space->_datapoint_to_object[reindexed_fv_index] = current_doc;
        vector_space->data.push_back(reindexed_fv);
      }

      // Update mapping from original terms to new indexes.
      std::map<std::string, int>::const_iterator mapping_it;
      for (mapping_it = term_to_index.begin();
          mapping_it != term_to_index.end(); ++mapping_it) {
        vector_space->_coordinate_mapping[mapping_it->second] =
            mapping_it->first;
      }

      return vector_space;
    }
};

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
