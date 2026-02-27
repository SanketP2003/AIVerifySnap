"use client";

import { motion } from "framer-motion";
import { ArrowRight, ShieldCheck, Cpu, Fingerprint } from "lucide-react";
import Link from "next/link";
import { UploadDropzone } from "@/components/shared/UploadDropzone";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();

  const handleUpload = () => {
    router.push("/detect");
  };

  return (
    <div className="flex flex-col items-center justify-center bg-background min-h-screen pt-10 relative">
      {/* Ambient background lights (Sarvam style) */}
      <div className="subtle-blue-light top-0 left-[-200px]" />
      <div className="subtle-orange-light top-[400px] right-[-200px]" />

      {/* Hero Section */}
      <section className="relative w-full px-4 sm:px-6 lg:px-8 py-20 lg:py-32 flex flex-col items-center justify-center overflow-hidden">

        {/* Soft Spotlight matching Sarvam aesthetic */}
        <div className="absolute top-10 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-[#e7af55]/5 rounded-full blur-[120px] -z-10 pointer-events-none" />

        <div className="text-center space-y-6 max-w-[900px] mx-auto z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="mb-8">
              <span className="inline-flex items-center rounded-full border border-black/5 bg-white/50 px-4 py-1.5 text-sm font-medium text-[#131313] backdrop-blur-sm">
                India&apos;s Sovereign Verification Platform
              </span>
            </div>
            <h1 className="text-6xl md:text-8xl font-serif text-[#131313] text-balance leading-[1.1] mb-2 tracking-tight">
              Verify media. <br />
              <span className="text-black/80 font-serif">Detect deepfakes.</span>
            </h1>
          </motion.div>

          <motion.p
            className="text-xl md:text-2xl text-muted-foreground text-balance max-w-2xl mx-auto font-medium"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          >
            Built on advanced forensic networks. Powered by frontier-class explainability models. Delivering irrefutable authenticity.
          </motion.p>

          <motion.div
            className="flex items-center justify-center gap-4 pt-4"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Link href="/detect" className="px-8 py-4 bg-foreground text-background font-semibold rounded-full hover:bg-foreground/90 transition-transform active:scale-95 text-lg">
              Start Verification
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Main Upload / Drag Area with Glass Look */}
      <section className="w-full max-w-5xl mx-auto px-4 pb-32">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="relative rounded-[2rem] md:rounded-[3rem] p-4 md:p-8 bg-card border shadow-2xl overflow-hidden glass-panel"
        >
          <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent pointer-events-none" />
          <UploadDropzone onUpload={handleUpload} />
        </motion.div>
      </section>

      {/* Pillars Section */}
      <section className="w-full py-24 bg-muted/30 border-t">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="mb-20 md:text-center md:max-w-3xl md:mx-auto">
            <h2 className="text-4xl md:text-5xl font-bold tracking-tighter">Powering an Authentic Future</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-12 lg:gap-16">
            {[
              {
                icon: Cpu,
                title: "Forensic by design",
                desc: "Run multi-layer deep feature analysis with full transparency, utilizing state-of-the-art Error Level Analysis networks."
              },
              {
                icon: ShieldCheck,
                title: "State-of-the-art Models",
                desc: "Industry-leading vision models built to capture the most subtle synthetic compression artifacts and generative flaws."
              },
              {
                icon: Fingerprint,
                title: "Protection at the core",
                desc: "Scan population-scale datasets to identify unauthorized likeness use and automate digital identity takedown workflows."
              }
            ].map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: i * 0.1 }}
                className="flex flex-col space-y-4"
              >
                <div className="w-12 h-12 flex items-center justify-center rounded-2xl bg-foreground text-background mb-4">
                  <feature.icon className="w-6 h-6" />
                </div>
                <h3 className="text-2xl font-bold tracking-tight">{feature.title}</h3>
                <p className="text-lg text-muted-foreground leading-relaxed">{feature.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="w-full py-32 overflow-hidden relative">
        <div className="max-w-5xl mx-auto px-4 flex flex-col items-center text-center space-y-8 relative z-10">
          <h2 className="text-5xl md:text-7xl font-black tracking-tighter max-w-3xl text-balance">
            Build an authentic future with AIVerifySnap.
          </h2>
          <Link href="/protection" className="flex items-center gap-2 text-xl font-medium text-muted-foreground hover:text-foreground transition-colors group">
            Explore Identity Protection <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </Link>
        </div>
        {/* Huge Background text like Sarvam */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full overflow-hidden flex justify-center opacity-[0.03] pointer-events-none select-none -z-10">
          <span className="text-[15rem] font-black tracking-tighter whitespace-nowrap">AIVERIFYSNAP</span>
        </div>
      </section>

    </div>
  );
}
