import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { MotionConfig } from "motion/react"
import { ThemeProvider } from "next-themes"

import App from "@/App"
import { AudioPlayerProvider } from "@/components/audio-player"
import { Toaster } from "@/components/ui/sonner"
import { TooltipProvider } from "@/components/ui/tooltip"
import "@/index.css"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      <MotionConfig reducedMotion="user">
        <TooltipProvider delayDuration={400}>
          <AudioPlayerProvider>
            <App />
            <Toaster position="bottom-right" />
          </AudioPlayerProvider>
        </TooltipProvider>
      </MotionConfig>
    </ThemeProvider>
  </StrictMode>,
)
