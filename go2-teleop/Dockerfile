FROM --platform=$BUILDPLATFORM golang:1.27.2 AS build
ARG TARGETARCH
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY backend.go main.go index.html ./
COPY internal ./internal
RUN CGO_ENABLED=0 GOOS=linux GOARCH=$TARGETARCH go build -mod=readonly -trimpath -ldflags='-s -w' -o /go2-teleop .

FROM scratch
COPY --from=build /go2-teleop /go2-teleop
COPY LICENSE /licenses/go2-teleop-LICENSE
COPY internal/agentpb/LICENSE /licenses/WendyOS-LICENSE
COPY internal/agentpb/NOTICE /licenses/WendyOS-NOTICE
EXPOSE 8882
ENTRYPOINT ["/go2-teleop"]
CMD ["--listen", "0.0.0.0:8882"]
