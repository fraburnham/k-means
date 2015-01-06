(ns k-means.core)

(defn sum [f & rest]
  (reduce +' (apply (partial map f) rest)))

(defn sq-diff [a b]
  (Math/pow (- a b) 2))

(defn sq-euclidean-distance [a b]
  (sum sq-diff a b))

(defn cost [points point-assignments]
  (/ (sum sq-euclidean-distance points point-assignments)
     (count points)))

(defn random-centroids [k points]
  (repeatedly k #(nth points (rand-int (count points)))))

(defn cluster-assignment [points centroids]
  (map (fn [point] ;for each point
         (first ;get rid of the distance measurement
           (reduce (fn [r centroid]
                     ;if this centroid is closer, return it
                     (let [[_ last-distance] r
                           distance (sq-euclidean-distance point centroid)]
                       (if (< distance last-distance) [centroid distance] r)))
                   centroids)))
       points))

(defn move-centroids [points point-assignments centroids]
  (map (fn [c]
         (let [[sum count] (reduce
                             (fn [r point]
                               (if (nil? point)
                                 r
                                 (let [[sum count] r]
                                   [(map + sum point) (inc count)])))
                             (map (fn [point point-assignment]
                                    (if (= point-assignment c) point nil))
                                  points point-assignments))]
           (map / sum count))) ;this should be modified to detect centroids with 0 points
       centroids))

;convergence-fn compares new-centroids to old-centroids
;to determine if the centroids are placed optimally
(defn learn
  ([k points] (learn k points =))
  ([k points convergence-fn]
    (loop [centroids (random-centroids k points)]
      (let [point-assignments (cluster-assignment points centroids)
            new-centroids (move-centroids points point-assignments centroids)]
        (if (convergence-fn new-centroids centroids)
          centroids
          (recur new-centroids))))))

;data is the set of features that we would like to classify
(defn classify [new-point centroids]
  ;find the closest centroid and return that as the classification
  (first (cluster-assignment [new-point] centroids)))
