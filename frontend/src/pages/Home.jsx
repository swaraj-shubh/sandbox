import React from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function Home() {
  return (
    <div className="min-h-screen bg-black text-[#00ff99] relative overflow-hidden">
      {/* Neon grid background */}
      <div className="absolute inset-0 z-0">
        <div className="w-full h-full bg-[radial-gradient(circle_at_center,_rgba(0,255,100,0.05)_0%,_black_80%)]"></div>
        <div className="absolute inset-0 opacity-10 bg-[linear-gradient(90deg,rgba(0,255,100,0.2)_1px,transparent_1px),linear-gradient(rgba(0,255,100,0.2)_1px,transparent_1px)] bg-[size:30px_30px]" />
      </div>

      {/* Flickering text animation */}
      <style>
        {`
        @keyframes flicker {
          0%, 18%, 22%, 25%, 53%, 57%, 100% {
            opacity: 1;
          }
          20%, 24%, 55% {
            opacity: 0.4;
          }
        }
        @keyframes scan {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100%); }
        }
        `}
      </style>

      {/* Green scanning line */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden z-0 pointer-events-none">
        <div className="w-full h-1 bg-[#00ff99]/20 animate-[scan_4s_linear_infinite]"></div>
      </div>

      {/* Content */}
      <div className="relative z-10 flex flex-col items-center justify-center text-center h-screen px-4">
        <h1
          className="text-5xl md:text-6xl font-extrabold tracking-widest text-[#00ff99]
          drop-shadow-[0_0_15px_#00ff99] animate-[flicker_2s_infinite]"
        >
          VAJRA
        </h1>
        <p className="mt-4 text-[#00ff99cc] max-w-xl text-lg font-mono leading-relaxed">
          A Secure LLM Proxy Gateway Against Prompt Injection
        </p>

        <div className="mt-10 flex flex-wrap gap-4 justify-center">
          <Link to="/auth">
            <Button
              className="bg-[#00ff99] text-black font-semibold hover:bg-[#00e688] hover:scale-105 transition transform shadow-[0_0_15px_#00ff99]"
            >
              ACCESS TERMINAL
            </Button>
          </Link>
          <Link to="#docs">
            <Button
              variant="outline"
              className="border-[#00ff99] text-[#00ff99] hover:bg-[#00ff9915] hover:text-[#00ffcc]"
            >
              VIEW PROTOCOLS
            </Button>
          </Link>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-5xl">
          {[
            { title: "XYZ", desc: "XXYYZZ." },
            { title: "XYZ", desc: "XXYYZZ." },
            { title: "XYZ", desc: "XXYYZZ." },
          ].map((f) => (
            <Card
              key={f.title}
              className="bg-[#001a0d]/60 border border-[#00ff99]/10 backdrop-blur-md shadow-[0_0_25px_#00ff9940] hover:shadow-[0_0_35px_#00ff99a0] transition"
            >
              <CardContent className="p-6">
                <h3 className="text-xl font-semibold text-[#00ffcc] mb-2">{f.title}</h3>
                <p className="text-sm text-[#00ff99bb] font-mono">{f.desc}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Footer */}
      <footer className="absolute bottom-0 w-full py-4 text-center text-xs text-[#00ff99aa] font-mono border-t border-[#00ff99]/10">
        [ SYSTEM STATUS: ONLINE ] • Uptime: 99.999% • Version: v1.0.0 • Classified Access Only
      </footer>
    </div>
  );
}
