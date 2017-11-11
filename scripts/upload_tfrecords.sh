#!/usr/bin/env bash

mkdir -p data/tfrecords
rm data/tfrecords/*

mv train eval data/tfrecords/

gsutil -m cp data/tfrecords/* gs://cnn-test/data/tfrecords/