import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import HowItWorks from "./components/HowItWorks";
import Demo from "./components/Demo";
import Features from "./components/Features";
import Footer from "./components/Footer";

export default function App() {
  return (
    <div id="top" className="app-bg relative min-h-screen text-slate-900 dark:text-slate-100">
      {/* Animated mesh-gradient backdrop (fixed, behind everything) */}
      <div className="mesh-bg" aria-hidden="true">
        <div className="grain" />
      </div>

      {/* All real content sits above the mesh */}
      <div className="relative z-10">
        <Navbar />
        <main>
          <Hero />
          <HowItWorks />
          <Demo />
          <Features />
        </main>
        <Footer />
      </div>
    </div>
  );
}
