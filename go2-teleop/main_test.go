package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
)

const testKey = "offline-fixture-key-not-a-credential-0123456789"

type recordingBackend struct {
	mu            sync.Mutex
	opens, closes int
	frames        []command
}

func (b *recordingBackend) Mode() string { return fakeBackend{}.Mode() }
func (b *recordingBackend) Open(ctx context.Context) (controller, error) {
	b.mu.Lock()
	b.opens++
	b.mu.Unlock()
	return &recordingController{fakeController: fakeController{ready: ready{Challenge: nonce(), TimeoutMS: 350, Mode: b.Mode()}}, owner: b}, nil
}

type recordingController struct {
	fakeController
	owner *recordingBackend
}

func (c *recordingController) Exchange(ctx context.Context, f command) (ready, error) {
	c.owner.mu.Lock()
	c.owner.frames = append(c.owner.frames, f)
	c.owner.mu.Unlock()
	return c.fakeController.Exchange(ctx, f)
}
func (c *recordingController) Close() {
	c.owner.mu.Lock()
	c.owner.closes++
	c.owner.mu.Unlock()
	c.fakeController.Close()
}
func loginFixture(t *testing.T, server *httptest.Server) *http.Cookie {
	t.Helper()
	r, _ := http.NewRequest("POST", server.URL+"/api/session", strings.NewReader(`{"key":"`+testKey+`"}`))
	r.Header.Set("Origin", server.URL)
	response, err := server.Client().Do(r)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != 204 {
		t.Fatalf("login status %d", response.StatusCode)
	}
	return response.Cookies()[0]
}
func browserFixture(t *testing.T) (*httptest.Server, *recordingBackend, *http.Cookie) {
	t.Helper()
	b := &recordingBackend{}
	server := httptest.NewServer(newWebServer(b, testKey, false))
	t.Cleanup(server.Close)
	return server, b, loginFixture(t, server)
}
func wsFixture(t *testing.T, server *httptest.Server, cookie *http.Cookie) *websocket.Conn {
	t.Helper()
	headers := http.Header{"Origin": {server.URL}, "Cookie": {cookie.String()}}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	t.Cleanup(cancel)
	ws, _, err := websocket.Dial(ctx, server.URL+"/api/control", &websocket.DialOptions{HTTPHeader: headers})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ws.CloseNow() })
	return ws
}
func readReady(t *testing.T, ws *websocket.Conn) ready {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_, raw, err := ws.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var r ready
	if json.Unmarshal(raw, &r) != nil {
		t.Fatal("invalid ready")
	}
	return r
}
func sendFrame(t *testing.T, ws *websocket.Conn, f command) {
	t.Helper()
	raw, _ := json.Marshal(f)
	if err := ws.Write(context.Background(), websocket.MessageText, raw); err != nil {
		t.Fatal(err)
	}
}
func waitClosed(t *testing.T, b *recordingBackend) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		b.mu.Lock()
		closed := b.closes > 0
		b.mu.Unlock()
		if closed {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("controller did not close within timeout")
}
func TestPageAndHealthNeverOpenRobot(t *testing.T) {
	server, b, cookie := browserFixture(t)
	for _, path := range []string{"/", "/api/status"} {
		r, _ := http.NewRequest("GET", server.URL+path, nil)
		r.AddCookie(cookie)
		response, err := server.Client().Do(r)
		if err != nil {
			t.Fatal(err)
		}
		response.Body.Close()
		if response.StatusCode != 200 {
			t.Fatal(response.StatusCode)
		}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.opens != 0 {
		t.Fatal("passive page/health opened robot")
	}
}
func TestLoginAndControlRejectMissingKeyAndCrossOrigin(t *testing.T) {
	server, b, cookie := browserFixture(t)
	for _, origin := range []string{"", "https://wrong.example"} {
		r, _ := http.NewRequest("POST", server.URL+"/api/session", strings.NewReader(`{"key":"`+testKey+`"}`))
		r.Header.Set("Origin", origin)
		response, err := server.Client().Do(r)
		if err != nil {
			t.Fatal(err)
		}
		response.Body.Close()
		if response.StatusCode != 403 {
			t.Fatal("accepted cross-origin login")
		}
	}
	for _, headers := range []http.Header{{"Origin": {server.URL}}, {"Origin": {"https://wrong.example"}, "Cookie": {cookie.String()}}} {
		_, response, err := websocket.Dial(context.Background(), server.URL+"/api/control", &websocket.DialOptions{HTTPHeader: headers})
		if err == nil {
			t.Fatal("accepted unauthorized websocket")
		}
		response.Body.Close()
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.opens != 0 {
		t.Fatal("unauthorized request opened backend")
	}
}
func TestBrowserHoldReleaseStop(t *testing.T) {
	server, b, cookie := browserFixture(t)
	ws := wsFixture(t, server, cookie)
	r := readReady(t, ws)
	sendFrame(t, ws, command{Challenge: r.Challenge, Forward: .15})
	r = readReady(t, ws)
	sendFrame(t, ws, command{Challenge: r.Challenge})
	r = readReady(t, ws)
	sendFrame(t, ws, command{Challenge: r.Challenge, Stop: true})
	waitClosed(t, b)
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.frames) != 3 || b.frames[0].Forward != .15 || b.frames[1].Forward != 0 || !b.frames[2].Stop {
		t.Fatalf("unexpected frames %+v", b.frames)
	}
}
func TestBrowserFailuresCloseController(t *testing.T) {
	for _, name := range []string{"expired", "disconnect", "replay", "speed", "malformed"} {
		t.Run(name, func(t *testing.T) {
			server, b, cookie := browserFixture(t)
			ws := wsFixture(t, server, cookie)
			r := readReady(t, ws)
			switch name {
			case "expired":
			case "disconnect":
				ws.CloseNow()
			case "replay":
				sendFrame(t, ws, command{Challenge: "old", Forward: .1})
			case "speed":
				sendFrame(t, ws, command{Challenge: r.Challenge, Forward: 1})
			case "malformed":
				_ = ws.Write(context.Background(), websocket.MessageText, []byte(`{"forward":"bad"}`))
			}
			waitClosed(t, b)
			b.mu.Lock()
			defer b.mu.Unlock()
			if len(b.frames) != 0 {
				t.Fatal("invalid/expired browser input forwarded to robot")
			}
		})
	}
}
func TestBrowserSingleControllerAndManualReenable(t *testing.T) {
	server, b, cookie := browserFixture(t)
	first := wsFixture(t, server, cookie)
	_ = readReady(t, first)
	headers := http.Header{"Origin": {server.URL}, "Cookie": {cookie.String()}}
	_, response, err := websocket.Dial(context.Background(), server.URL+"/api/control", &websocket.DialOptions{HTTPHeader: headers})
	if err == nil || response.StatusCode != 409 {
		t.Fatal("accepted second controller")
	}
	response.Body.Close()
	first.CloseNow()
	waitClosed(t, b)
	second := wsFixture(t, server, cookie)
	_ = readReady(t, second)
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.opens != 2 {
		t.Fatal("re-enable did not open a fresh controller")
	}
}
func TestAgentBackendRejectsTCPAndMissingSerial(t *testing.T) {
	for _, path := range []string{"", "127.0.0.1:50051", "unix:///tmp/agent.sock"} {
		if _, conn, err := newAgentBackend(path, "GO2TEST1"); err == nil {
			conn.Close()
			t.Fatalf("accepted %q", path)
		}
	}
	if _, conn, err := newAgentBackend("/tmp/agent.sock", ""); err == nil {
		conn.Close()
		t.Fatal("accepted missing serial")
	}
}
