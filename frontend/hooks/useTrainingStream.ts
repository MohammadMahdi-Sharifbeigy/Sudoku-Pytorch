"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const MAX_RETRIES = 3;

export type DatasetMode = "mnist_fonts" | "mnist_hoda" | "all";
export type TrainingStatus = "idle" | "connecting" | "training" | "complete" | "error";

export interface TrainingConfig {
  epochs: number;
  learningRate: number;
  batchSize: number;
  datasetMode: DatasetMode;
}

export interface EpochData {
  epoch: number;
  epochs: number;
  train_loss: number;
  val_loss: number;
  train_acc: number;
  val_acc: number;
}

export interface TestResults {
  test_loss: number;
  test_acc: number;
  y_true: number[];
  y_pred: number[];
}

export interface TrainingMetrics {
  bestValLoss: number | null;
  bestEpoch: number | null;
  testLoss: number | null;
  testAcc: number | null;
  modelPath: string | null;
  retryCount: number;
  status: TrainingStatus;
}

type StreamEvent =
  | ({ type: "epoch" } & EpochData)
  | { type: "best_model"; val_loss: number; epoch: number }
  | ({ type: "test_results" } & TestResults)
  | { type: "complete"; model_path: string }
  | { type: "error"; message: string };

export function useTrainingStream() {
  const [status, setStatus] = useState<TrainingStatus>("idle");
  const [epochData, setEpochData] = useState<EpochData[]>([]);
  const [bestValLoss, setBestValLoss] = useState<number | null>(null);
  const [bestEpoch, setBestEpoch] = useState<number | null>(null);
  const [testResults, setTestResults] = useState<TestResults | null>(null);
  const [modelPath, setModelPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  const eventSourceRef = useRef<EventSource | null>(null);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const completeRef = useRef(false);
  const trainingUrlRef = useRef<string | null>(null);
  const retryCountRef = useRef(0);
  const reconnectRef = useRef<(url: string) => void>(() => undefined);

  const closeStream = useCallback(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;

    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
  }, []);

  const handleEvent = useCallback((event: StreamEvent) => {
    switch (event.type) {
      case "epoch":
        setStatus("training");
        setEpochData((current) => [...current, event]);
        break;
      case "best_model":
        setBestValLoss(event.val_loss);
        setBestEpoch(event.epoch);
        break;
      case "test_results":
        setTestResults(event);
        break;
      case "complete":
        completeRef.current = true;
        setModelPath(event.model_path);
        setStatus("complete");
        closeStream();
        break;
      case "error":
        completeRef.current = true;
        setError(event.message);
        setStatus("error");
        closeStream();
        break;
    }
  }, [closeStream]);

  const openStream = useCallback((url: string) => {
    const source = new EventSource(url);
    eventSourceRef.current = source;

    source.onopen = () => {
      setError(null);
      setStatus((current) => (current === "connecting" ? "training" : current));
    };

    source.onmessage = (message) => {
      try {
        handleEvent(JSON.parse(message.data) as StreamEvent);
      } catch {
        completeRef.current = true;
        setError("Training stream returned an unreadable event.");
        setStatus("error");
        closeStream();
      }
    };

    source.onerror = () => {
      source.close();

      if (completeRef.current) return;

      if (retryCountRef.current >= MAX_RETRIES) {
        setError("Training stream connection dropped. Reconnect limit reached.");
        setStatus("error");
        eventSourceRef.current = null;
        return;
      }

      retryCountRef.current += 1;
      setRetryCount(retryCountRef.current);
      setStatus("connecting");

      retryTimerRef.current = setTimeout(() => {
        if (trainingUrlRef.current && !completeRef.current) {
          reconnectRef.current(trainingUrlRef.current);
        }
      }, 900 * 2 ** (retryCountRef.current - 1));
    };
  }, [closeStream, handleEvent]);

  useEffect(() => {
    reconnectRef.current = openStream;
  }, [openStream]);

  const startTraining = useCallback((config: TrainingConfig) => {
    closeStream();
    completeRef.current = false;
    retryCountRef.current = 0;
    setRetryCount(0);
    setStatus("connecting");
    setError(null);
    setEpochData([]);
    setBestValLoss(null);
    setBestEpoch(null);
    setTestResults(null);
    setModelPath(null);

    const url = new URL(`${API_BASE}/api/train/stream`);
    url.searchParams.set("epochs", String(config.epochs));
    url.searchParams.set("learning_rate", String(config.learningRate));
    url.searchParams.set("batch_size", String(config.batchSize));
    url.searchParams.set("dataset_mode", config.datasetMode);

    trainingUrlRef.current = url.toString();
    openStream(url.toString());
  }, [closeStream, openStream]);

  const resetTraining = useCallback(() => {
    closeStream();
    completeRef.current = true;
    setStatus("idle");
    setError(null);
    setRetryCount(0);
  }, [closeStream]);

  useEffect(() => closeStream, [closeStream]);

  const metrics = useMemo<TrainingMetrics>(() => ({
    bestValLoss,
    bestEpoch,
    testLoss: testResults?.test_loss ?? null,
    testAcc: testResults?.test_acc ?? null,
    modelPath,
    retryCount,
    status,
  }), [bestEpoch, bestValLoss, modelPath, retryCount, status, testResults]);

  return {
    isTraining: status === "connecting" || status === "training",
    epochData,
    metrics,
    error,
    startTraining,
    resetTraining,
    testResults,
    status,
  };
}
