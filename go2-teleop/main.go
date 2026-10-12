package main

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/coder/websocket"
)

//go:embed index.html
var page embed.FS

type webServer struct {
	backend       backend
	keyHash       [32]byte
	cookie        string
	secureCookie  bool
	mu            sync.Mutex
	active        bool
	loginAttempts int
	loginWindow   time.Time
}

func newWebServer(b backend, key string, secure bool) *webServer {
	return &webServer{backend: b, keyHash: sha256.Sum256([]byte(key)), cookie: nonce(), secureCookie: secure}
}
func sameOrigin(r *http.Request) bool {
	u, err := url.Parse(r.Header.Get("Origin"))
	return err == nil && (u.Scheme == "http" || u.Scheme == "https") && u.Host == r.Host && u.User == nil && u.Path == "" && u.RawQuery == "" && u.Fragment == ""
}
func (s *webServer) authenticated(r *http.Request) bool {
	c, err := r.Cookie("go2_teleop")
	return err == nil && subtle.ConstantTimeCompare([]byte(c.Value), []byte(s.cookie)) == 1
}
func (s *webServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Referrer-Policy", "no-referrer")
	w.Header().Set("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
	switch {
	case r.Method == "GET" && r.URL.Path == "/":
		body, _ := page.ReadFile("index.html")
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(body)
	case r.Method == "POST" && r.URL.Path == "/api/session":
		if !sameOrigin(r) {
			http.Error(w, "same-origin requests required", 403)
			return
		}
		s.mu.Lock()
		if time.Since(s.loginWindow) > time.Minute {
			s.loginWindow = time.Now()
			s.loginAttempts = 0
		}
		s.loginAttempts++
		limited := s.loginAttempts > 10
		s.mu.Unlock()
		if limited {
			http.Error(w, "too many login attempts; wait a minute", 429)
			return
		}
		var body struct {
			Key string `json:"key"`
		}
		d := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1024))
		d.DisallowUnknownFields()
		if d.Decode(&body) != nil {
			http.Error(w, "invalid login", 400)
			return
		}
		hash := sha256.Sum256([]byte(body.Key))
		if subtle.ConstantTimeCompare(hash[:], s.keyHash[:]) != 1 {
			http.Error(w, "invalid access key", 401)
			return
		}
		http.SetCookie(w, &http.Cookie{Name: "go2_teleop", Value: s.cookie, Path: "/", HttpOnly: true, Secure: s.secureCookie, SameSite: http.SameSiteStrictMode})
		w.WriteHeader(204)
	case r.Method == "GET" && r.URL.Path == "/api/status":
		if !s.authenticated(r) {
			http.Error(w, "login required", 401)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"mode": s.backend.Mode()})
	case r.Method == "GET" && r.URL.Path == "/api/control":
		if !s.authenticated(r) || !sameOrigin(r) {
			http.Error(w, "authenticated same-origin control required", 403)
			return
		}
		s.mu.Lock()
		if s.active {
			s.mu.Unlock()
			http.Error(w, "another controller is active", 409)
			return
		}
		s.active = true
		s.mu.Unlock()
		defer func() { s.mu.Lock(); s.active = false; s.mu.Unlock() }()
		ws, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer ws.CloseNow()
		ws.SetReadLimit(1024)
		ctx, cancel := context.WithCancel(r.Context())
		defer cancel()
		// Opening the backend is explicit Enable, never a page load/health check.
		control, err := s.backend.Open(ctx)
		if err != nil {
			_ = ws.Close(websocket.StatusPolicyViolation, "control unavailable; check agent configuration")
			return
		}
		defer control.Close()
		next := control.Initial()
		for {
			// Local expiry covers BOTH reply delivery and the next browser input.
			budget := time.Duration(next.TimeoutMS) * time.Millisecond
			if budget <= 0 || budget > inputBudget {
				return
			}
			inputDeadline := time.Now().Add(budget)
			inputCtx, inputCancel := context.WithDeadline(ctx, inputDeadline)
			raw, _ := json.Marshal(next)
			err = ws.Write(inputCtx, websocket.MessageText, raw)
			var message []byte
			var kind websocket.MessageType
			if err == nil {
				kind, message, err = ws.Read(inputCtx)
			}
			inputCancel()
			if err != nil || !time.Now().Before(inputDeadline) {
				return
			}
			var f command
			decoder := json.NewDecoder(strings.NewReader(string(message)))
			decoder.DisallowUnknownFields()
			if kind != websocket.MessageText || decoder.Decode(&f) != nil || decoder.Decode(&struct{}{}) != io.EOF || f.Challenge != next.Challenge || !bounded(f) {
				return
			}
			if !time.Now().Before(inputDeadline) {
				return
			}
			commandCtx, commandCancel := context.WithTimeout(ctx, 250*time.Millisecond)
			next, err = control.Exchange(commandCtx, f)
			commandCancel()
			if err != nil {
				return
			}
		}
	default:
		http.NotFound(w, r)
	}
}

func main() {
	listen := flag.String("listen", "127.0.0.1:8882", "HTTP bind address; remote access needs an approved private tunnel")
	hardware := flag.Bool("allow-motion", false, "use real robot control; requires separate supervised physical approval")
	serial := flag.String("serial", "", "exact owner-confirmed paired Go2 serial (hardware only)")
	socket := flag.String("agent-socket", os.Getenv("WENDY_AGENT_SOCKET"), "private agent Unix socket (never a TCP address)")
	tokenFile := flag.String("token-file", "", "private file containing a random access key (32+ characters)")
	secure := flag.Bool("secure-cookie", true, "require HTTPS; false only for localhost or an approved encrypted private mesh")
	flag.Parse()
	key := os.Getenv("GO2_TELEOP_TOKEN")
	if *tokenFile != "" {
		info, err := os.Lstat(*tokenFile)
		if err != nil || !info.Mode().IsRegular() || info.Mode().Perm()&0077 != 0 || info.Size() > 1024 {
			fmt.Fprintln(os.Stderr, "access key file must be a private regular file <=1024 bytes")
			os.Exit(1)
		}
		raw, err := os.ReadFile(*tokenFile)
		if err != nil {
			fmt.Fprintln(os.Stderr, "cannot read access key file")
			os.Exit(1)
		}
		key = strings.TrimSpace(string(raw))
	}
	if len(key) < 32 || len(key) > 512 {
		fmt.Fprintln(os.Stderr, "provide a random 32+ character access key using GO2_TELEOP_TOKEN or --token-file")
		os.Exit(1)
	}
	var b backend = fakeBackend{}
	if *hardware {
		real, conn, err := newAgentBackend(*socket, *serial)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer conn.Close()
		b = real
	}
	handler := newWebServer(b, key, *secure)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	server := &http.Server{Addr: *listen, Handler: handler, ReadHeaderTimeout: 3 * time.Second, IdleTimeout: 30 * time.Second, BaseContext: func(_ net.Listener) context.Context { return ctx }}
	go func() {
		<-ctx.Done()
		closeCtx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_ = server.Shutdown(closeCtx)
	}()
	fmt.Fprintf(os.Stderr, "Go2 teleop: %s; listening on %s; disabled until browser Enable\n", b.Mode(), *listen)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		fmt.Fprintln(os.Stderr, "web server failed")
		os.Exit(1)
	}
}
