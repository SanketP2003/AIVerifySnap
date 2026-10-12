"use client";

import { AlertCircle, ExternalLink, Loader2, ShieldCheck, Check } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

interface PlatformReportCardProps {
  platform: string;
  url: string;
  matchConfidence: number;
  dateDetected: string;
}

export function PlatformReportCard({
  platform,
  url,
  matchConfidence,
  dateDetected,
}: PlatformReportCardProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [reported, setReported] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleTakedown = () => {
    setIsProcessing(true);
    // Simulate API call for takedown
    setTimeout(() => {
      setIsProcessing(false);
      setReported(true);
      toast.success(`Takedown notice generated for ${platform}`, {
        description: "DMCA request package prepared and forwarded to platform abuse desk.",
      });
    }, 1800);
  };

  const handleCopyUrl = () => {
    navigator.clipboard.writeText(url);
    setCopied(true);
    toast.info("Infringing URL copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  const formattedUrl = url.startsWith("http") ? url : `https://${url}`;

  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between p-5 rounded-2xl border border-border bg-card/60 hover:bg-card/90 transition-all gap-4">
      <div className="flex flex-col flex-1 items-start gap-1.5 w-full">
        <div className="flex flex-wrap items-center gap-2">
          <AlertCircle className="w-5 h-5 text-destructive shrink-0" />
          <h3 className="font-semibold text-lg text-foreground">{platform}</h3>
          <span className="text-xs bg-destructive/10 text-destructive font-bold px-2.5 py-0.5 rounded-full border border-destructive/20">
            {matchConfidence}% Match
          </span>
        </div>

        <div className="flex items-center gap-2 max-w-full text-sm text-muted-foreground">
          <a
            href={formattedUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="truncate max-w-[260px] sm:max-w-md hover:text-primary transition-colors hover:underline inline-flex items-center gap-1"
            title={formattedUrl}
          >
            <span>{url}</span>
            <ExternalLink className="w-3.5 h-3.5 shrink-0" />
          </a>
          <button
            type="button"
            onClick={handleCopyUrl}
            aria-label="Copy infringement URL"
            className="p-1 hover:bg-muted rounded-md text-muted-foreground hover:text-foreground transition-colors shrink-0"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-green-500" /> : <span className="text-[11px] font-mono">Copy</span>}
          </button>
        </div>

        <span className="text-xs text-muted-foreground/80 mt-1">Detected: {dateDetected}</span>
      </div>

      <div className="w-full sm:w-auto shrink-0">
        {!reported ? (
          <button
            onClick={handleTakedown}
            disabled={isProcessing}
            className="w-full sm:w-auto px-5 py-2.5 bg-primary text-primary-foreground font-semibold rounded-xl hover:bg-primary/90 active:scale-95 transition-all flex items-center justify-center gap-2 disabled:opacity-50 text-sm shadow-sm"
          >
            {isProcessing && <Loader2 className="w-4 h-4 animate-spin" />}
            {isProcessing ? "Generating Notice..." : "Generate Takedown Notice"}
          </button>
        ) : (
          <div className="w-full sm:w-auto px-5 py-2.5 bg-green-500/10 text-green-600 font-semibold rounded-xl flex items-center justify-center gap-2 border border-green-500/20 text-sm cursor-default">
            <ShieldCheck className="w-4 h-4" />
            Notice Dispatched
          </div>
        )}
      </div>
    </div>
  );
}
