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
    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between p-4 sm:p-5 rounded-xl border border-border bg-card hover:border-foreground/20 transition-all gap-4 shadow-sm">
      <div className="flex flex-col flex-1 items-start gap-1.5 w-full">
        <div className="flex flex-wrap items-center gap-2">
          <AlertCircle className="w-4 h-4 text-destructive shrink-0" />
          <h3 className="font-bold text-base text-foreground font-sans">{platform}</h3>
          <span className="text-[11px] font-mono bg-destructive/10 text-destructive font-bold px-2 py-0.5 rounded-md border border-destructive/25">
            {matchConfidence}% Match
          </span>
        </div>

        <div className="flex items-center gap-2 max-w-full text-xs text-muted-foreground">
          <a
            href={formattedUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="truncate max-w-[260px] sm:max-w-md hover:text-foreground transition-colors hover:underline inline-flex items-center gap-1 font-mono"
            title={formattedUrl}
          >
            <span>{url}</span>
            <ExternalLink className="w-3 h-3 shrink-0" />
          </a>
          <button
            type="button"
            onClick={handleCopyUrl}
            aria-label="Copy infringement URL"
            className="px-1.5 py-0.5 hover:bg-muted rounded text-[11px] font-mono text-muted-foreground hover:text-foreground transition-colors shrink-0 border border-border/50"
          >
            {copied ? <Check className="w-3 h-3 text-green-500 inline mr-0.5" /> : null}
            {copied ? "Copied" : "Copy"}
          </button>
        </div>

        <span className="text-[11px] font-mono text-muted-foreground/80 mt-0.5">Detected: {dateDetected}</span>
      </div>

      <div className="w-full sm:w-auto shrink-0">
        {!reported ? (
          <button
            onClick={handleTakedown}
            disabled={isProcessing}
            className="w-full sm:w-auto px-4 py-2 bg-primary text-primary-foreground font-semibold rounded-lg hover:bg-primary/90 active:scale-98 transition-all flex items-center justify-center gap-1.5 disabled:opacity-50 text-xs shadow-sm"
          >
            {isProcessing && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
            <span>{isProcessing ? "Generating Notice..." : "Generate DMCA Notice"}</span>
          </button>
        ) : (
          <div className="w-full sm:w-auto px-4 py-2 bg-green-500/10 text-green-600 dark:text-green-400 font-semibold rounded-lg flex items-center justify-center gap-1.5 border border-green-500/30 text-xs cursor-default">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Notice Dispatched</span>
          </div>
        )}
      </div>
    </div>
  );
}
