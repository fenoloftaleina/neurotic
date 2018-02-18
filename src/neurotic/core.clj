(ns neurotic.core
  (:require [clojure.pprint :refer [pprint]]
            [clojure.string :as string]))

(def activation-fn (fn [x] (Math/tanh x)))
(def dactivation-fn (fn [y] (- 1.0 (* y y))))

(def data
  [{:input [0 0] :expected-result [0]}
   {:input [0 1] :expected-result [1]}
   {:input [1 0] :expected-result [1]}
   {:input [1 1] :expected-result [2]}
   {:input [1 2] :expected-result [3]}
   {:input [2 0] :expected-result [2]}
   {:input [0 2] :expected-result [2]}
   {:input [2 1] :expected-result [3]}
   {:input [1 2] :expected-result [3]}
   {:input [2 2] :expected-result [4]}])

(def weights
  [[[0.2 0.4] [0.3 0.6] [0.5 0.8]] [[0.5 0.7 0.9]]])

(defn feed-forward [input weights]
  (loop [input input
         weights weights
         new-net []]
    (if-let [layer-weights (first weights)]
      (let [new-neurons
            (mapv
              activation-fn
              (mapv
                #(reduce + (map * input %))
                layer-weights))]
        (recur
          new-neurons
          (rest weights)
          (conj new-net new-neurons)))
      new-net)))

(def input-neurons [1 2])

(def new-neurons
  (feed-forward input-neurons weights))

(prn new-neurons)

(defn output-deltas [outputs targets]
  (mapv
    (fn [output target]
      (*
       (dactivation-fn output)
       (- target output)))
    outputs targets))

(def new-output-deltas
  (output-deltas
    (last new-neurons)
    [3]))

(prn new-output-deltas)
(prn (last weights))

(defn layer-deltas [output-deltas neurons output-weights]
  (mapv
    *
    (mapv dactivation-fn neurons)
    (reduce
      (fn [a e]
        (mapv + a e))
      (mapv
        (fn [weights delta]
          (mapv
            (partial * delta)
            weights))
        output-weights output-deltas))))

(def new-layer-deltas
  (layer-deltas
    new-output-deltas
    (first new-neurons)
    (last weights)))

(prn new-layer-deltas)

(def learning-rate 0.5)

(defn backpropagate [neurons weights deltas learning-rate]
  (mapv
    (fn [layer-neurons layer-weights layer-deltas]
      (mapv
        (fn [weights delta]
          (mapv
            (fn [weight neuron]
              (+ weight (* learning-rate delta neuron)))
            weights layer-neurons))
        layer-weights layer-deltas))
    neurons weights deltas))

;; [i i]
;; [[w w] [w w] [w w]]
;; [d d d]

(backpropagate
  (cons input-neurons (butlast new-neurons))
  weights
  [new-layer-deltas new-output-deltas]
  learning-rate)

(defn train-single [weights {:keys [input expected-result]} learning-rate]
  (let [neurons (feed-forward input weights)
        output-deltas (output-deltas (last neurons) expected-result)
        deltas
        (reduce
          (fn [deltas [layer-neurons layer-weights]]
            (cons (layer-deltas (first deltas) layer-neurons layer-weights)
                  deltas))
          [output-deltas]
          (map vector
               (reverse (butlast neurons))
               (butlast (reverse weights))))]
    (backpropagate (cons input (butlast neurons)) weights deltas learning-rate)))

(pprint weights)
(pprint (train-single weights (nth data 1) 0.5))

(defn train [weights data learning-rate]
  (reduce (fn [current-weights data-point]
            (train-single current-weights data-point learning-rate))
          weights
          data))

(def a (train weights data 0.5))
(def b (train (train weights data 0.5) data 0.5))

(defn results-errors [weights data]
  (map (fn [{:keys [input expected-result] :as data-row}]
         (let [result (feed-forward input weights)
               error (mapv - expected-result (last result))]
           (assoc data-row :result result :error error)))
       data))

(pprint (results-errors a data))

(defn errors [results-errors]
  (map
    (fn [{:keys [error expected-result]}]
      (map
        (fn [e e-r]
          (str e
               " "
               (try
                 (-> e (+ e-r) (/ e) (* 100))
                 (catch Exception e "NaN"))
               "%"))
        error expected-result))
    results-errors))

(prn (errors (results-errors weights data)))
(prn (errors (results-errors a data)))
(prn (errors (results-errors b data)))

;; (defn train-until [errors-threshold weights data learning-rate]
;;   (loop [weights weights
;;          recent-errors (map :expected-result data)]
;;     (prn recent-errors)
;;     (prn weights)
;;     (if (not (some #(>= % errors-threshold) (flatten recent-errors)))
;;       weights
;;       (let [new-weights (train-batch weights data learning-rate)
;;             new-errors (errors (see new-weights data))]
;;         (recur new-weights new-errors)))))
;;
;; (train-until 0.5 weights data 0.5)

;; (def divisor 1000)
(def divisor Integer/MAX_VALUE)

(defn normalized [data]
  (let [n-fn #(double (/ % divisor))
        map-n-fn #(map n-fn %)]
    (map
      (fn [data-point]
        (-> data-point
            (update :input map-n-fn)
            (update :expected-result map-n-fn)))
      data)))

(defn randomized [weights]
  (map
    (fn [layer-weights]
      (map
        (fn [neuron-weights]
          (map
            (fn [weight]
              (rand))
            neuron-weights))
        layer-weights))
    weights))

(def new-rand-weights (randomized weights))

(defn see [weights data]
  (pprint
    (string/join
      " "
      (flatten
        (map
          (fn [{:keys [result expected-result]}]
            (map
              (fn [r e-r]
                (format "%.5f %.1f," (* r divisor) (* (double e-r) divisor)))
              (last result) expected-result))
          (results-errors weights data))))))

;; (def learned (atom [(train new-rand-weights data 0.1)]))
;; (defn more [n]
;;   (last @learned))
;; (more 1)

(defn hey [input weights]
  (->> (feed-forward (map #(double (/ % divisor)) input) weights)
       last
       (map (partial * divisor))))

(defn n-to-m [n m]
  (normalized
    (flatten (map (fn [a] (map (fn [b]
                                 {:input [a b] :expected-result [(+ a b)]})
                               (range n m))) (range n m)))))

(defn n-to-m-x [n m x]
  (flatten (repeat x (n-to-m n m))))

(def fifteen (train new-rand-weights (n-to-m-x 0 15 1) 0.2))
(def fifteen-2 (train fifteen (n-to-m-x 0 15 1) 0.2))
(def fifteen-5 (train fifteen-2 (n-to-m-x 0 15 3) 0.2))
(see fifteen (n-to-m 0 15))
(see fifteen-2 (n-to-m 0 15))
(see fifteen-5 (n-to-m 0 15))

(def fifteen-50 (train fifteen-5 (n-to-m-x 0 15 45) 0.2))
(see fifteen-50 (n-to-m 0 15))

(hey [90 100] fifteen-50)

(hey [27 50] (train new-rand-weights (n-to-m-x 0 10 1) 0.2))

;; weights->net

(def net (randomized [[[0.2 0.4] [0.3 0.6] [0.5 0.8]]
                      [[0.5 0.7 0.9] [0.5 0.7 0.9]]
                      [[1 1]]]))

(pprint net)
(pprint (train net (n-to-m-x 2 5 100) 0.2))

(def trained (train net (n-to-m-x 2 3 1) 0.2))

(hey [5 3] trained)

(see trained (n-to-m 0 10))
