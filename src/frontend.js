"use client";
import { useState } from "react";

export default function StoryApp() {
  const [formData, setFormData] = useState({
    title: "",
    setting: "",
    mainCharacter: "",
    genre: "",
    tone: "",
  });
  const [synopsis, setSynopsis] = useState("");
  const [chapters, setChapters] = useState([]);
  const [currentChapter, setCurrentChapter] = useState(null);
  const [question, setQuestion] = useState(null);
  const [answer, setAnswer] = useState("");

  const handleChange = (e) =>
    setFormData({ ...formData, [e.target.name]: e.target.value });

  async function handleSubmit(e) {
    e.preventDefault();
    const res = await fetch("http://localhost:8000/api/synopsis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    });
    const data = await res.json();
    setSynopsis(data.synopsis);
  }

  async function generateChapter() {
    setCurrentChapter({ number: chapters.length + 1, text: "" });
    const res = await fetch("http://localhost:8000/api/chapter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chapter: chapters.length + 1 }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      try {
        const messages = buffer.split("\n").filter(Boolean);
        for (let msg of messages) {
          const data = JSON.parse(msg);

          if (data.type === "text") {
            setCurrentChapter((prev) => ({
              ...prev,
              text: prev.text + data.content,
            }));
          } else if (data.type === "question") {
            setQuestion(data.content);
          } else if (data.type === "chapter_complete") {
            setChapters((prev) => [...prev, currentChapter]);
            setCurrentChapter(null);
          }
        }
        buffer = "";
      } catch {
        // wait for full JSON
      }
    }
  }

  async function submitAnswer() {
    const res = await fetch("http://localhost:8000/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer }),
    });
    setQuestion(null);
    setAnswer("");
  }

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">Interactive Story Generator</h1>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="space-y-2">
        {["title", "setting", "mainCharacter", "genre", "tone"].map((f) => (
          <input
            key={f}
            name={f}
            placeholder={f}
            value={formData[f]}
            onChange={handleChange}
            className="w-full p-2 border rounded"
          />
        ))}
        <button className="px-4 py-2 bg-blue-600 text-white rounded">
          Generate Synopsis
        </button>
      </form>

      {/* Synopsis */}
      {synopsis && (
        <div className="p-4 border rounded bg-gray-50">
          <h2 className="font-semibold">Story Synopsis</h2>
          <p>{synopsis}</p>
        </div>
      )}

      {/* Chapter Generator */}
      {synopsis && !currentChapter && (
        <button
          onClick={generateChapter}
          className="px-4 py-2 bg-green-600 text-white rounded"
        >
          Generate Chapter
        </button>
      )}

      {/* Current Chapter */}
      {currentChapter && (
        <div className="p-4 border rounded bg-gray-100">
          <h2 className="font-semibold">
            Chapter {currentChapter.number}
          </h2>
          <textarea
            value={currentChapter.text}
            readOnly
            rows={10}
            className="w-full p-2 border rounded mt-2"
          />
        </div>
      )}

      {/* Question */}
      {question && (
        <div className="p-4 border rounded bg-yellow-100">
          <p className="font-semibold">Question: {question}</p>
          <input
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            placeholder="Your answer"
            className="w-full p-2 border rounded mt-2"
          />
          <button
            onClick={submitAnswer}
            className="mt-2 px-4 py-2 bg-blue-600 text-white rounded"
          >
            Submit Answer
          </button>
        </div>
      )}

      {/* Past Chapters */}
      {chapters.map((ch, i) => (
        <div
          key={i}
          className="p-4 border rounded bg-white shadow mt-4"
        >
          <h2 className="font-semibold">Chapter {ch.number}</h2>
          <pre className="whitespace-pre-wrap">{ch.text}</pre>
        </div>
      ))}
    </div>
  );
}
